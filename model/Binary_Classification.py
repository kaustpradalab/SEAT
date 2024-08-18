
import numpy as np
import json
import os
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
import wandb
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
import time

from attention.model.modules.Decoder import AttnDecoder, FrozenAttnDecoder, PretrainedWeightsDecoder
from attention.model.modules.Encoder import Encoder
from attention.utlis.metrics import batch_tvd

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence
from attention.utlis.utlis import topk_overlap_loss,topK_overlap_true_loss


file_name = os.path.abspath(__file__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model():
    def __init__(self, configuration, args, pre_embed=None):
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        if args.encoder != "bert":
            configuration['model']['encoder']['pre_embed'] = pre_embed
        self.args=args

        self.encoder = Encoder.from_params(
            Params(configuration['model']['encoder'])).to(device)

        self.frozen_attn = args.frozen_attn
        # self.trian_mode = args.ours
        self.pre_loaded_attn = args.pre_loaded_attn
        self.trian_mode = args.train_mode

        configuration['model']['decoder'][
            'hidden_size'] = self.encoder.output_size
        if self.frozen_attn:
            self.decoder = FrozenAttnDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)
        elif self.pre_loaded_attn:
            self.decoder = PretrainedWeightsDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)
        else:
            self.decoder = AttnDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        if not self.frozen_attn:
            self.attn_params = list([
                v for k, v in self.decoder.named_parameters()
                if 'attention' in k
            ])
        self.decoder_params = list([
            v for k, v in self.decoder.named_parameters()
            if 'attention' not in k
        ])

        self.bsize = args.bsize

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        # lr = 2e-5 if args.encoder == "bert" else 0.001
        lr = self.args.lr
        self.encoder_optim = torch.optim.Adam(
            self.encoder_params,
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True)
        if not self.frozen_attn:
            self.attn_optim = torch.optim.Adam(
                self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(
            self.decoder_params,
            lr=0.001,
            weight_decay=weight_decay,
            amsgrad=True)

        pos_weight = configuration['training'].get(
            'pos_weight', [1.0] * self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)

        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2


        self.K = args.K
        self.topk_prox_metric = args.topk_prox_metric

        if args.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

    def target_model(self,embedd, data, decoder, encoder):
        encoder(data, revise_embedding=embedd)
        decoder(data=data)
        return torch.sigmoid(data.predict)

    @classmethod
    def init_from_config(cls, dirname, args, **kwargs):
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config, args)
        obj.load_values(dirname)
        return obj
    
    def eval_sens(self, dataset, X_PGDer=None,args=None,train=False):
        self.encoder.train()
        self.decoder.train()

        from attention.utlis.utlis import AverageMeter
        N = len(dataset)
        bsize = self.bsize
        batches = list(range(0, N, bsize))
        sens_auc = AverageMeter()
        
        for n in tqdm(batches):
        # for idx, batch_doc in enumerate(loader):
            batch_doc = dataset[n:n + bsize]
            if n+bsize > N:
                continue
            if len(batch_doc['label']) !=bsize:
                print(len(batch_doc['label']))
            assert len(batch_doc['label']) == bsize
            batch_data = BatchHolder(batch_doc, bert=args.encoder=="bert")
            # old prediction
            self.encoder(batch_data)
            self.decoder(batch_data)

            # old att pred
            old_pred = torch.sigmoid(batch_data.predict)
            old_att = batch_data.attn
            
            for ri in [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]:
                X_PGDer.radius = ri
                target_model = self.target_model
                def crit(gt, pred):
                    return batch_tvd(gt, pred)
                # PGD generate the new hidden
                assert batch_data.embedding.shape[0] == bsize
                assert batch_data.hidden.shape[0] == bsize
                loss = X_PGDer.perturb_return_loss(criterion=crit, x=batch_data.embedding.to(device), data=batch_data \
                                            , decoder=self.decoder, encoder=self.encoder,
                                            y=batch_data.label,
                                            target_model=target_model)
                sens_auc.update(loss * ri/2/bsize, bsize)

            batch_data.to_cpu()
            del batch_data
        return sens_auc.average()
    
    
    def eval_comp_suff(self, dataset, X_PGDer=None,args=None,train=False):
        self.encoder.eval()
        self.decoder.eval()

        from attention.utlis.utlis import AverageMeter
        N = len(dataset)
        bsize = self.bsize 
        batches = list(range(0, N, bsize*4))
        comp = AverageMeter()
        suff = AverageMeter()
        r_number = 5
        # eval_iter = 
        
        for i, n in tqdm(enumerate(batches)):
        # for idx, batch_doc in enumerate(loader):
            # if i == 3:
            #     break
            batch_doc = dataset[n:n + bsize]
            if n+bsize > N:
                continue
            if len(batch_doc['label']) !=bsize:
                print(len(batch_doc['label']))
            assert len(batch_doc['label']) == bsize
            batch_data = BatchHolder(batch_doc, bert=args.encoder=="bert")
            # old prediction
            self.encoder(batch_data)
            self.decoder(batch_data)

            # old att pred
            old_pred = torch.sigmoid(batch_data.predict)
            old_att = batch_data.attn.cpu()
            
            embed_comp = batch_data.embedding.cpu().clone()
            embed_suff = torch.zeros_like(batch_data.embedding).cpu()
            for k in range(r_number):
                idx_max_att = torch.argmax(old_att, dim=1)
                old_att[:,idx_max_att] = 0
                embed_comp[:, idx_max_att, :] = 0
                embed_suff[:, idx_max_att, :] = batch_data.embedding[:, idx_max_att, :].cpu().data
            
            # for comp
            self.encoder(batch_data, revise_embedding=embed_comp.to(device))
            self.decoder(data=batch_data)
            new_pred = torch.sigmoid(batch_data.predict)
            comp.update(batch_tvd(old_pred, new_pred)/bsize, bsize)
            del embed_comp
            
            # for suff
            self.encoder(batch_data, revise_embedding=embed_suff.to(device))
            self.decoder(data=batch_data)
            new_pred = torch.sigmoid(batch_data.predict)
            suff.update(batch_tvd(old_pred, new_pred)/bsize, bsize)
            del embed_suff
            batch_data.to_cpu()
            del batch_doc 
            del batch_data
            torch.cuda.empty_cache() 
            time.sleep(0.2)
        return comp.average(), suff.average()
    
    
    def perturb_x_eval(self, dataset, X_PGDer=None,args=None,train=False):
        """
        perturb the embedding and evaluate the tvd of model output change and jsd of attention weights change
        
        """
        
        self.encoder.train()
        self.decoder.train()

        from attention.utlis.utlis import AverageMeter
        N = len(dataset)
        bsize = self.bsize
        # print("bsize", bsize)
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        px_l1_att_diff = AverageMeter()
        px_l2_att_diff = AverageMeter()
        px_tvd_pred_diff = AverageMeter()
        px_jsd_att_diff = AverageMeter()
        
        for n in tqdm(batches):
            batch_doc = dataset[n:n + bsize]
            if n+bsize > N:
                continue
            if len(batch_doc['label']) !=bsize:
                print(len(batch_doc['label']))
            assert len(batch_doc['label']) == bsize
            batch_data = BatchHolder(batch_doc, bert=args.encoder=="bert")

            # get the hidden
            def crit(gt, pred):
                return batch_tvd(gt, pred)

            # old prediction
            self.encoder(batch_data)
            self.decoder(batch_data)

            # old att pred
            old_pred = torch.sigmoid(batch_data.predict)
            old_att = batch_data.attn

            target_model = self.target_model

            # PGD generate the new hidden
            assert batch_data.embedding.shape[0] == bsize
            assert batch_data.hidden.shape[0] == bsize
            new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding.to(device), data=batch_data \
                                         , decoder=self.decoder, encoder=self.encoder,
                                         y=batch_data.label,
                                         target_model=target_model)

            # pgd perturb the hidden
            self.encoder(batch_data, revise_embedding=new_embedd)
            self.decoder(data=batch_data)

            # diff of att
            # new_att = torch.sigmoid(batch_data.attn) 
            new_att = batch_data.attn
            px_l1_att_diff.update(torch.abs(old_att - new_att).mean().item(), len(batch_doc))
            px_l2_att_diff.update(torch.pow(old_att - new_att, 2).mean().item(), len(batch_doc))
            new_pred = torch.sigmoid(batch_data.predict)
            px_tvd_pred_diff.update(self.criterion(new_pred,old_pred).sum().item()/bsize,len(batch_doc))
            px_jsd_att_diff.update(js_divergence(old_att, new_att).squeeze(1).cpu().data.numpy().mean(), len(batch_doc))

            # batch_data
            batch_data.to_cpu()
            del batch_data
            # del batch_target_pred
            
        import numpy as np
        res = {
            "px_l1_att_diff":px_l1_att_diff.average(),
            "px_l2_att_diff":px_l2_att_diff.average(),
            "px_tvd_pred_diff":px_tvd_pred_diff.average(),
            "px_jsd_att_diff":px_jsd_att_diff.average(),
            "stab": px_tvd_pred_diff.average(),
        }
        return res



    def train_ours(self,dataset,
              PGDer=None,train=True,preturb_x=False,X_PGDer=None):
        self.encoder.train()
        self.decoder.train()


        from attention.utlis.utlis import AverageMeter
        loss_weighted = 0
        loss_unweighted = 0
        benign_loss_mean = 0
        adv_loss_mean = 0
        topk_true_loss_mean = 0
        topk_loss_mean = 0

        N = len(dataset)
        bsize = self.bsize
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)


        px_l1_att_diff = AverageMeter()
        px_l2_att_diff = AverageMeter()
        px_tvd_pred_diff = AverageMeter()
        px_jsd_att_diff = AverageMeter()


        for n in tqdm(batches):
            batch_doc = dataset[n:n + bsize]
            if n+bsize > N:
                continue
            batch_data = BatchHolder(batch_doc,bert=self.args.encoder == "bert")

            if preturb_x:
                def target_model(embedd, data, decoder,encoder):
                    encoder(data,revise_embedding=embedd)
                    decoder(data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(gt,pred)

                # old prediction
                self.encoder(batch_data)
                self.decoder(batch_data)

                # old att pred
                old_pred = torch.sigmoid(batch_data.predict)
                old_att = batch_data.attn
                
                new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data, decoder=self.decoder,encoder=self.encoder, y=batch_data.label,target_model=target_model)

                # pgd perturb the hidden
                self.encoder(batch_data,revise_embedding=new_embedd)
                self.decoder(data=batch_data)

                new_att = batch_data.attn
                px_l1_att_diff.update(torch.abs(old_att - new_att).mean().item(), len(batch_doc))
                px_l2_att_diff.update(torch.pow(old_att - new_att, 2).mean().item(), len(batch_doc))
                new_pred = torch.sigmoid(batch_data.predict)
                px_tvd_pred_diff.update(batch_tvd(old_pred, new_pred).item()/bsize, len(batch_doc))
                px_jsd_att_diff.update(
                    js_divergence(old_att, new_att).squeeze(
                        1).cpu().data.numpy().mean(), len(batch_doc))

            # to the true att and embedding
            self.encoder(batch_data)
            self.decoder(batch_data)
            
            # old_out = batch_data.predict.clone()
            old_att = batch_data.attn.clone()

            batch_target = batch_data.label.to(device)
            # batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1:  #(B, )
                batch_target = batch_target.unsqueeze(-1)  #(B, 1)
                
            benign_loss = self.criterion(batch_data.predict, batch_data.label)
            if self.args.method == "ours":
                def target_model(embedd, data, decoder,encoder):
                    encoder(data,revise_embedding=embedd)
                    decoder(data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(gt,pred)
                
                # we will perturb  e(x) to ensure robustness of models
                new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data \
                                        , decoder=self.decoder,encoder=self.encoder, y=batch_data.label,
                                        target_model=target_model)

                # pgd perturb the hidden
                self.encoder(batch_data,revise_embedding=new_embedd)
                self.decoder(data=batch_data)
            
                new_out = batch_data.predict
                new_att = batch_data.attn
                
                adv_loss = self.criterion(new_out, batch_data.label)
                
                topk_loss = topk_overlap_loss(new_att,
                                            old_att,K=self.K, metric=self.topk_prox_metric).mean()
                topk_true_loss = topK_overlap_true_loss(new_att,
                                            old_att,K=self.K)
                
                adv_loss = self.lambda_1 * adv_loss
                topk_loss = self.lambda_2 * topk_loss
            elif self.args.method == "word-at":
                def target_model(embedd, data, decoder,encoder):
                    encoder(data,revise_embedding=embedd)
                    decoder(data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(gt,pred)
                
                # we will perturb  e(x) to ensure robustness of models
                new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data \
                                        , decoder=self.decoder,encoder=self.encoder, y=batch_data.label,
                                        target_model=target_model)

                # pgd perturb the hidden
                self.encoder(batch_data,revise_embedding=new_embedd)
                self.decoder(data=batch_data)
            
                new_out = batch_data.predict
                new_att = batch_data.attn
                adv_loss = self.criterion(new_out, batch_data.label)
            
                adv_loss = self.lambda_1 * adv_loss
            elif self.args.method == "word-iat":
                def target_model(embedd, data, decoder,encoder):
                    encoder(data,revise_embedding=embedd)
                    decoder(data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(gt,pred)
                
                # we will perturb  e(x) to ensure robustness of models
                new_embedd = X_PGDer.perturb_iat(criterion=crit, x=batch_data.embedding, data=batch_data \
                                        , decoder=self.decoder,encoder=self.encoder, y=batch_data.label,
                                        target_model=target_model)

                # pgd perturb the hidden
                self.encoder(batch_data,revise_embedding=new_embedd)
                self.decoder(data=batch_data)
            
                new_out = batch_data.predict
                # new_att = batch_data.attn
                adv_loss = self.criterion(new_out, batch_data.label)
                adv_loss = self.lambda_1 * adv_loss
            elif self.args.method == "attention-at":
                ## perturb attention weight
                def target_model(w, data, decoder, encoder):
                    decoder(revise_att=w, data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(pred, gt)

                # PGD generate the new weight
                new_att = PGDer.perturb(criterion=crit, x=batch_data.attn, data=batch_data \
                                        , decoder=self.decoder, y=batch_data.label,
                                        target_model=target_model)
                
                # output the prediction tvd of new weight and old weight
                self.decoder(batch_data, revise_att=new_att)
                
                new_out = batch_data.predict
                adv_loss = self.criterion(new_out, batch_data.label)
                adv_loss = self.lambda_1 * adv_loss
            elif self.args.method == "attention-iat":
                ## perturb attention weight
                def target_model(w, data, decoder, encoder):
                    decoder(revise_att=w, data=data)
                    return torch.sigmoid(data.predict)
                    
                def crit(gt, pred):
                    return batch_tvd(pred, gt)
                    
                # PGD generate the new weight
                new_att = PGDer.perturb_iat(criterion=crit, x=batch_data.attn.unsqueeze(2), data=batch_data \
                                        , decoder=self.decoder, y=batch_data.label,
                                        target_model=target_model)
                
                if len(new_att.shape) == 3:
                    new_att = new_att.squeeze(2)
                # output the prediction tvd of new weight and old weight
                self.decoder(batch_data, revise_att=new_att)
                
                new_out = batch_data.predict
                adv_loss = self.criterion(new_out, batch_data.label)
                adv_loss = self.lambda_1 * adv_loss
            elif self.args.method == "attention-rp":
                ## perturb attention weight
                def target_model(w, data, decoder, encoder):
                    decoder(revise_att=w, data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(pred, gt)

                # PGD generate the new weight
                new_att = PGDer.perturb_random(criterion=crit, x=batch_data.attn, data=batch_data \
                                        , decoder=self.decoder, y=batch_data.label,
                                        target_model=target_model)
                
                # output the prediction tvd of new weight and old weight
                self.decoder(batch_data, revise_att=new_att)
                
                new_out = batch_data.predict
                adv_loss = self.criterion(new_out, batch_data.label)
                adv_loss = self.lambda_1 * adv_loss
                
                
            loss_orig = benign_loss + adv_loss
            
            if self.args.method == "ours":
                loss_orig += topk_loss
                
            weight = batch_target * self.pos_weight + (1 - batch_target)
            loss = (loss_orig * weight).mean(1).sum()
            

            if train:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                if not self.frozen_attn:
                    self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                if not self.frozen_attn:
                    self.attn_optim.step()

            loss_weighted += float(loss.data.cpu().item())
            loss_unweighted += float(loss_orig.data.cpu().mean().item())
            benign_loss_mean += float(benign_loss.data.cpu().mean().item())
            adv_loss_mean += float(
                adv_loss.data.cpu().mean().item())
            
            if self.args.method == "ours":
                topk_loss_mean += float(topk_loss.data.cpu().mean().item())
                topk_true_loss_mean += float(topk_true_loss)
            
            batch_data.to_cpu()
            del batch_data

        res = {
            "loss_weighted": loss_weighted,
            "loss_unweighted": loss_unweighted,
            "benign_loss": benign_loss_mean,
            "adv_loss": adv_loss_mean,}
        
        if self.args.method == "ours":
            res["true_topk_loss"] = topk_true_loss_mean
            res['topk_loss'] = topk_loss_mean
        
        if preturb_x:
            res["px_tvd_pred_diff"] = px_tvd_pred_diff.average()
            res["px_jsd_att_diff"] = px_jsd_att_diff.average()
            res["px_l1_att_diff"] = px_l1_att_diff.average()
            res["px_l2_att_diff"] = px_l2_att_diff.average()
            
        return res

    def train(self,
              data,
              train=True,
              PDGer=None,
              args=None):
        
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            
        bsize = self.bsize
        N = len(data)
        loss_total = 0
        loss_unweighted = 0
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)


        for n in tqdm(batches):
            batch_doc = data[n:n + bsize]
            batch_data = BatchHolder(batch_doc,bert=args.encoder=="bert")

            self.encoder(batch_data)
            self.decoder(batch_data)
            assert args.train_mode == 'std_train'
            ## standard trianing
            loss_orig = self.criterion(batch_data.predict, batch_data.label)
            
            weight = batch_data.label * self.pos_weight + (1 - batch_data.label)
            loss = (loss_orig * weight).mean(1).sum()
            loss_orig = loss_orig.mean(1).sum()

            if train:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                if not self.frozen_attn:
                    self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                if not self.frozen_attn:
                    self.attn_optim.step()

            loss_total += float(loss.data.cpu().item())

            assert args.train_mode  == "std_train"
            loss_unweighted += float(loss_orig.data.cpu().sum().item())

        assert args.train_mode  == "std_train"
        return loss_total * bsize / N, loss_total, loss_unweighted
    
    def evaluate(self, data,args=None):
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        assert bsize == args.bsize
        N = len(data)

        outputs = []
        attns = []
        js_scores = []

        for n in tqdm(range(0, N, bsize)):
            batch_doc = data[n:n + bsize]
            if n+bsize > N:
                continue
            batch_data = BatchHolder(batch_doc,bert=args.encoder=="bert")

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            if args.use_attention:  #and n == 0:
                attn = batch_data.attn.cpu().data.numpy()  #.astype('float16')
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy(
            )  #.astype('float16')
            outputs.append(predict)
            
            del batch_data

        outputs = [x for y in outputs for x in y]
        if args.use_attention:
            attns = [x for y in attns for x in y]
        return outputs, attns

    def save_values(self, use_dirname=None, save_model=True):
        """
        this function is to save the config and model ckpt
        :param use_dirname:
        :type use_dirname:
        :param save_model:
        :type save_model:
        :return:
        :rtype:
        """
        if use_dirname is not None:
            dirname = use_dirname
        else:
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model:
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')
            self.dirname = dirname

        return dirname

    def end_clean(self):
        import os
        os.remove(self.dirname + '/enc.th')
        os.remove(self.dirname + '/dec.th')

    def load_values(self, dirname):
        # self.dirname = dirname
        self.encoder.load_state_dict(
            # torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
            torch.load(dirname + '/enc.th',map_location={'cuda:1': 'cuda:0'}),strict=False,)
        self.decoder.load_state_dict(
            # torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))
            torch.load(dirname + '/dec.th',map_location={'cuda:1': 'cuda:0'}), strict=False,)