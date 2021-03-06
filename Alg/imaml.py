import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import copy

from Alg.support import GBML
from loss import bce_dice_loss,dice_coef
from utils import apply_grad, mix_grad

class iMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.lamb =100
        self.n_cg =1
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel(train_input)
        inner_loss =  bce_dice_loss(train_logit, train_target)
        diffopt.step(inner_loss)

        return None

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.network.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        vec = []
        for param in hv:
            vec.append(param.reshape(-1))
        hv=torch.cat(vec).detach()
       # hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrixparameters
        return hv/self.lamb + x

    def outer_loop(self, batch, is_train):

        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        dice_log = 0
        grad_list = []
        loss_list = []

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):

                for step in range(self.args.n_inner):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                train_logit = fmodel(train_input)
                in_loss =  bce_dice_loss(train_logit, train_target)

                test_logit = fmodel(test_input)

                outer_loss = bce_dice_loss(test_logit, test_target)
                loss_log += outer_loss.item()/self.batch_size

                out_cut = np.copy(test_logit.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < 0.3)] = 0.0  #threshold
                out_cut[np.nonzero(out_cut >= 0.3)] = 1.0

                with torch.no_grad():
                    dice_log += dice_coef(out_cut, test_target.data.cpu().numpy()).item()/self.batch_size

                if is_train:
                    params = list(fmodel.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(outer_loss, params))
                    implicit_grad = self.cg(in_grad, outer_grad, params)
                    grad_list.append(implicit_grad)
                    loss_list.append(outer_loss.item())

        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()

            return loss_log, dice_log, grad_log
        else:
            return loss_log, dice_log
