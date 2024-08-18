import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type, ascending=True):
        self.radius = radius  # attack radius
        self.steps = steps # how many step to conduct pgd
        self.step_size = step_size  # coefficient of PGD
        self.random_start = random_start
        self.norm_type = norm_type # which norm of your noise
        self.ascending = ascending # perform gradient ascending, i.e, to maximum the loss

    def perturb_return_loss(self, criterion, x, data, decoder,y,target_model,encoder=None):
        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()

        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        # assert adv_x.shape[0] == 8

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        # model.eval()
        decoder.eval()
        for pp in decoder.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            _y = target_model(adv_x, data,decoder,encoder)
            loss = criterion(y.to(device), _y)
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)
        
        ''' reopen autograd of model after pgd '''
        # decoder.trian()
        for pp in decoder.parameters():
            pp.requires_grad = True
            
        with torch.no_grad():
            _y = target_model(adv_x, data,decoder,encoder)
            loss = criterion(y.to(device), _y)

        return loss.item()
    
    def perturb(self, criterion, x, data, decoder,y,target_model,encoder=None):
        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()

        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        # model.eval()
        decoder.eval()
        for pp in decoder.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            # print("adv_x:", adv_x.shape)
            # assert adv_x.shape[0] == 8
            _y = target_model(adv_x, data,decoder,encoder)
            loss = criterion(y.to(device), _y)
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)

        ''' reopen autograd of model after pgd '''
        # decoder.trian()
        for pp in decoder.parameters():
            pp.requires_grad = True

        return adv_x.data
    def perturb_random(self, criterion, x, data, decoder,y,target_model,encoder=None):
        if self.steps==0 or self.radius==0:
            return x.clone()
        adv_x = x.clone()
        if self.norm_type == 'l-infty':
            adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
        else:
            adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
        self._clip_(adv_x, x)
        return adv_x.data
    
    def perturb_iat(self, criterion, x, data, decoder,y,target_model,encoder=None):
        if self.steps==0 or self.radius==0:
            return x.clone()
        
        B = x.shape[0]
        L = x.shape[1]
        H = x.shape[2]
        nb_num = 8
        
        alpha = torch.rand(B,L,nb_num,1).to(device)
        import numpy as np
        A_1 = x.unsqueeze(2).expand(B,L,nb_num,H)
        A_2 = x.unsqueeze(1).expand(B,L,L,H)
        rand_idx = []
        for i in range(L):
            rand_idx.append(np.random.choice(L,nb_num,replace=False))
        rand_idx = np.array(rand_idx)
        rand_idx = torch.tensor(rand_idx).long().reshape(1,L,1,nb_num).expand(B,L,H,nb_num).to(device)
        # A_2 = A_2[:,np.arange(0,L), rand_idx,:]
        A_2 = torch.gather(A_2.reshape(B,L,H,L),-1,rand_idx).reshape(B,L,nb_num, H)
        A_e = A_1 - A_2
        # A_e
        # adv_x = (A_e * alpha).sum(dim=-1) + x.clone()
        
        adv_x = x.clone()

        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
        self._clip_(adv_x, x)

        # assert adv_x.shape[0] == 8

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        # model.eval()
        decoder.eval()
        for pp in decoder.parameters():
            pp.requires_grad = False
            
        adv_x = x.clone()
        
        alpha.requires_grad_()

        for step in range(self.steps):
            alpha.requires_grad_()
            dot_Ae_alpha = (A_e * alpha).sum(dim=-2)
            # print("dot_Ae_alpha:", dot_Ae_alpha.shape)
            
            adv_x.add_(torch.sign(dot_Ae_alpha), alpha=self.step_size)
            
            self._clip_(adv_x, x)
            
            if encoder is None:
                adv_x_input = adv_x.squeeze(-1)
            else:
                adv_x_input = adv_x
            
            _y = target_model(adv_x_input, data,decoder,encoder)
            loss = criterion(y.to(device), _y)
            grad = torch.autograd.grad(loss, [alpha],retain_graph=True)[0]
            # with torch.no_grad():
            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)
                assert self.norm_type == 'l-infty'
                alpha = alpha.detach()+ grad * 0.01

        ''' reopen autograd of model after pgd '''
        # decoder.trian()
        for pp in decoder.parameters():
            pp.requires_grad = True

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0],-1)**2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0],-1).abs().sum(dim=1)
            norm = norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(0, 1)
