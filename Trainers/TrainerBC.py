from attention.utlis.metrics import calc_metrics_classification, print_metrics
import attention.model.Binary_Classification as BC
import codecs, json
from tqdm import tqdm
import numpy as np
from attention.preprocess import vectorizer
import wandb
import torch


class Trainer():
    def __init__(self, dataset, args, config):
        Model = BC.Model
        self.bert = args.encoder == "bert"
        pred_embed = None if self.bert else dataset.vec.embeddings
        self.model = Model(config, args, pre_embed=pred_embed)
        self.metrics = calc_metrics_classification
        self.display_metrics = True
        self.PGDer = None

    def train_standard(self, data, args, save_on_metric='roc_auc'):

        best_metric = 0.0
        for i in tqdm(range(args.n_epoch)):
            _, loss_tr, loss_tr_orig = self.model.train(data['train'],args=args)
            _, loss_te, loss_te_orig = self.model.train(data['test'],args=args,train=False)
            
            predictions_tr, attentions_tr = self.model.evaluate(data['train'],args=args)
            predictions_tr = np.array(predictions_tr)
            train_metrics = self.metrics(np.array(data['train'].y[:predictions_tr.shape[0]]), predictions_tr)
            print_str = "FULL (WEIGHTED) LOSS: %f | ORIG (UNWEIGHTED) LOSS: %f" % (loss_tr, loss_tr_orig)
            print(print_str)

            print("TRAIN METRICS:")
            if self.display_metrics:
                print_metrics(train_metrics, adv=False)

            predictions_te, attentions_te = self.model.evaluate(data['test'], args=args)
            predictions_te = np.array(predictions_te)
            test_metrics = self.metrics(np.array(data['test'].y[:predictions_te.shape[0]]), predictions_te)
            
            
            wandb.log({
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "train_loss": loss_tr,
                "train_loss_unweighted": loss_tr_orig,
                "test_loss": loss_te,
                "test_loss_unweighted": loss_te_orig,
            })

            print("TEST METRICS:")
            if self.display_metrics:
                print_metrics(test_metrics, adv=False)

            metric = test_metrics[save_on_metric]
            if metric > best_metric:
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else:
                save_model = False
                print("Model not saved on ", save_on_metric, metric)

            dirname = self.model.save_values(save_model=save_model)

            if save_model:
                attentions_tr = [el.tolist() for el in attentions_tr]
                attentions_te = [el.tolist() for el in attentions_te]
                print("SAVING PREDICTIONS AND ATTENTIONS")
                json.dump(predictions_tr.tolist(),
                          codecs.open(dirname + '/train_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(predictions_te.tolist(),
                          codecs.open(dirname + '/test_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_tr,
                          codecs.open(dirname + '/train_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_te,
                          codecs.open(dirname + '/test_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)

            print("DIRECTORY:", dirname)

            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

    def train_ours(self, dataset, args=None):
        br = False
        n_fail = 0
        best_f1 = 0


        if args.eval_baseline:
            # for eval the original model
            evaluator = Evaluator(args.gold_label_dir, args)
            original_metric, _, _ = evaluator.evaluate(dataset['test'], save_results=False)
            #
            wandb.log({
                "original_metric": original_metric,
            })

            # log original performance of defense x preturb
            res_baseline = evaluator.model.perturb_x_eval(dataset['test'], X_PGDer=self.X_PGDer, args=args)
            
            ## three metrics
            comp, suff = evaluator.model.eval_comp_suff(dataset['test'], X_PGDer=self.X_PGDer, args=args)
            sens = evaluator.model.eval_sens(dataset['test'], X_PGDer=self.X_PGDer, args=args)
            
            wandb.log({
                "baseline_suff_te": suff,
                "baseline_comp_te": comp,
                "baseline_sens_te": sens,
            })

            for k,v in res_baseline.items():
                wandb.log({
                    "baseline_"+k+"_te": v,
                })
            
            del evaluator.model.encoder
            del evaluator.model.decoder
            del evaluator.model
            del evaluator
            torch.cuda.empty_cache() 
            print("waiting for gpu to clean up")
            import time
            time.sleep(10)
        
        print("training our model ",args.n_epoch, " epochs in total")
        for i in tqdm(range(args.n_epoch)):
            res_tr = self.model.train_ours(dataset['train'], PGDer=self.PGDer, preturb_x=False, X_PGDer=self.X_PGDer,train=True)

            for k,v in res_tr.items():
                wandb.log({
                    k+"_tr": v,
                })
            loss_tr = res_tr['loss_weighted']
            
            res_te = self.model.train_ours(
                dataset['test'],
                PGDer=self.PGDer, train=False, preturb_x=True, X_PGDer=self.X_PGDer)
            wandb.log({
                "px_l2_att_diff":res_te['px_l2_att_diff'],
                "px_l1_att_diff": res_te['px_l1_att_diff'],
                "px_tvd_pred_diff": res_te['px_tvd_pred_diff'],
                "px_jsd_att_diff":res_te["px_jsd_att_diff"],
                "stab": res_te['px_tvd_pred_diff'],
            })
            loss_te = res_te['loss_weighted']
            
            wandb.log({
                "train_loss":loss_tr,
                "test_loss":loss_te,
            })

            for k,v in res_te.items():
                wandb.log({
                    k+"_te": v,
                })

            predictions_tr, attentions_tr = self.model.evaluate(dataset['train'],args=args)

            predictions_tr = np.array(predictions_tr)
            train_metrics = self.metrics(np.array(dataset['train'].y[:predictions_tr.shape[0]]), predictions_tr)
            
            print("TRAIN METRICS:")
            if self.display_metrics:
                print_metrics(train_metrics, adv=True)
            #
            predictions_te, attentions_te = self.model.evaluate(dataset['test'], args=args)

            predictions_te = np.array(predictions_te)
            test_metrics = self.metrics(np.array(dataset['test'].y[:predictions_te.shape[0]]), predictions_te)

            wandb.log({
                "test_metrics": test_metrics,
                "train_metrics": train_metrics,
            })

            print("TEST METRICS:")
            if self.display_metrics:
                print_metrics(test_metrics, adv=True)

            

            if test_metrics['macro avg/f1-score'] > best_f1:
                best_f1 = test_metrics['macro avg/f1-score']
                n_fail = 0
                save_model = True
            else:
                n_fail += 1
                save_model = False
                # print("Model not saved on Training Loss: ", loss_tr)
                if n_fail >= 10:
                    br = True
                    
            dirname = self.model.save_values(save_model=save_model)
            if save_model:
                attentions_tr = [el.tolist() for el in attentions_tr]
                attentions_te = [el.tolist() for el in attentions_te]
                print("SAVING PREDICTIONS AND ATTENTIONS")
                json.dump(predictions_tr.tolist(),
                          codecs.open(dirname + '/train_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(predictions_te.tolist(),
                          codecs.open(dirname + '/test_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_tr,
                          codecs.open(dirname + '/train_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_te,
                          codecs.open(dirname + '/test_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)

            print("DIRECTORY:", dirname)

            if br:
                break

            if args.eval_baseline:
                wandb.log(
                    {
                        "att l2 decrease": res_baseline['px_l2_att_diff'] - res_te['px_l2_att_diff'],
                        "tvd_decrease": res_baseline['px_tvd_pred_diff'] - res_te['px_tvd_pred_diff'],
                        "att l1 decrease": res_baseline['px_l1_att_diff'] - res_te['px_l1_att_diff'],
                        "att l1 decrease ratio": (res_baseline['px_l1_att_diff'] - res_te['px_l1_att_diff']) / res_baseline['px_l1_att_diff'],
                        "att l2 decrease ratio": (res_baseline['px_l2_att_diff'] - res_te['px_l2_att_diff']) / res_baseline['px_l2_att_diff'],
                        "tvd_decrease ratio": (res_baseline['px_tvd_pred_diff'] - res_te['px_tvd_pred_diff']) / res_baseline['px_tvd_pred_diff'],
                        "att jsd decrease": res_baseline['px_jsd_att_diff'] - res_te['px_jsd_att_diff'],
                        "att jsd decrease ratio": (res_baseline['px_jsd_att_diff'] - res_te['px_jsd_att_diff']) / res_baseline['px_jsd_att_diff'],
                        'stab_decrease ratio':(res_baseline['px_tvd_pred_diff'] - res_te['px_tvd_pred_diff']) / res_baseline['px_tvd_pred_diff']
                    }
                )

        
class Evaluator():
    def __init__(self, dirname, args):
        Model = BC.Model
        self.model = Model.init_from_config(dirname, args)
        self.model.dirname = dirname
        self.metrics = calc_metrics_classification
        self.display_metrics = True
        self.trian_mode = args.train_mode
        self.args = args

    def evaluate(self, test_data, save_results=False):
        if self.args.train_mode == "adv_train":
            predictions, attentions = self.model.evaluate(test_data, args=self.args)
            predictions = np.array(predictions)
            test_metrics = self.metrics(np.array(test_data.y[:predictions.shape[0]]), predictions)
        else:
            predictions, attentions = self.model.evaluate(test_data, args=self.args)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y[:predictions.shape[0]], predictions)

        if save_results:
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions
        return test_metrics, predictions, attentions
