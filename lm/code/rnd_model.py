import sys
import math
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RND(nn.Module):
    def __init__(self, input_dim=100, proj_dim=100, 
                 rn_n_hidden_layers=1, rn_hidden_dim=200,
                 student_resnet_n_blocks=1, student_resnet_inner_dim=200,
                 enforce_lipschitz=False, per_layer_max_l1_op_norm=0.99,
                 use_layer_norm=False,
                 scaling_coefficient=-1.0):
        super(RND, self).__init__()
        self.input_dim=input_dim
        self.proj_dim=proj_dim
        self.rn_n_hidden_layers=rn_n_hidden_layers
        self.rn_hidden_dim=rn_hidden_dim
        self.student_resnet_n_blocks=student_resnet_n_blocks
        self.student_resnet_inner_dim=student_resnet_inner_dim
        self.enforce_lipschitz=enforce_lipschitz
        self.per_layer_max_l1_op_norm = torch.scalar_tensor(per_layer_max_l1_op_norm)
        self.use_layer_norm=use_layer_norm
        self.scaling_coefficient = torch.scalar_tensor(scaling_coefficient)
        assert((not enforce_lipschitz) or (student_resnet_n_blocks == 0)), "Lipschitz constant can't be enforced on Resnet blocks"
        assert(not(enforce_lipschitz and use_layer_norm)), "Lipschitz constant enforcement not supported with layer norm"
        
        rn_layers, student_base_layers = [], []
        prev_dim = input_dim
        for i in range(rn_n_hidden_layers):
            if use_layer_norm:
                rn_layers.append(nn.LayerNorm(prev_dim, elementwise_affine=False))
            rn_layers.append(nn.Linear(prev_dim, rn_hidden_dim, bias=False))
            torch.nn.init.kaiming_uniform_(rn_layers[-1].weight, a=0.01)
            self.project_weight_(rn_layers[-1].weight)
            rn_layers.append(nn.LeakyReLU())
            
            if use_layer_norm:
                student_base_layers.append(nn.LayerNorm(prev_dim, elementwise_affine=False))
            student_base_layers.append(nn.Linear(prev_dim, rn_hidden_dim, bias=False))
            torch.nn.init.kaiming_uniform_(student_base_layers[-1].weight, a=0.01)
            self.project_weight_(student_base_layers[-1].weight)
            student_base_layers.append(nn.LeakyReLU())
            prev_dim = rn_hidden_dim
        if use_layer_norm:
            rn_layers.append(nn.LayerNorm(prev_dim, elementwise_affine=False))
        rn_layers.append(nn.Linear(rn_hidden_dim, proj_dim, bias=False))
        self.project_weight_(rn_layers[-1].weight)
        if use_layer_norm:
            rn_layers.append(nn.LayerNorm(proj_dim, elementwise_affine=False))
            
        if use_layer_norm:
            student_base_layers.append(nn.LayerNorm(prev_dim, elementwise_affine=False))
        student_base_layers.append(nn.Linear(rn_hidden_dim, proj_dim, bias=False))
        self.project_weight_(student_base_layers[-1].weight)
        if use_layer_norm:
            student_base_layers.append(nn.LayerNorm(proj_dim, elementwise_affine=False))
        
        self.rn = nn.Sequential(*rn_layers)
        self.student_base = nn.Sequential(*student_base_layers)
        for param in self.rn.parameters():
            param.requires_grad=False
        
        student_resnet_blocks = []
        for i in range(student_resnet_n_blocks):
            resnet_block = nn.Sequential(
                nn.Linear(proj_dim, student_resnet_inner_dim, bias=False),
                nn.LeakyReLU(),
                nn.Linear(student_resnet_inner_dim, proj_dim, bias=False))
            torch.nn.init.xavier_uniform_(resnet_block[-1].weight, gain=0.1)
            student_resnet_blocks.append(resnet_block)
        self.student_resnet = nn.ModuleList(student_resnet_blocks)
        
        self.post_scaling_gain = nn.Parameter(torch.scalar_tensor(1.0))
        self.post_scaling_gain.requires_grad=False
    
    def forward(self, x, return_projs=False):
        rn_proj = self.rn(x)
        student_proj = self.student_base(x)
        for i in range(self.student_resnet_n_blocks):
            student_proj = student_proj + self.student_resnet[i](student_proj)
        diff = rn_proj - student_proj
        sq_error = (diff * diff).mean(dim=-1)
        rv = sq_error
        if return_projs:
            rv = (rv, rn_proj, student_proj)
        return rv
    
    def project_weight_(self, w):
        if not self.enforce_lipschitz:
            return
        with torch.no_grad():
            l1_op_norm = w.norm(p=1, dim=-1).max()
            if l1_op_norm > self.per_layer_max_l1_op_norm:
                new_w = w * self.per_layer_max_l1_op_norm / l1_op_norm
                w.set_(new_w)
    
    def project_student_(self):
        for i in range(self.rn_n_hidden_layers):
            cur_layer = self.student_base[i]
            if "weight" in vars(cur_layer):
                self.project_weight_(cur_layer.weight)
    
    def freeze_student(self, unfreeze=False):
        for param in chain(self.student_base.parameters(), self.student_resnet.parameters()):
          param.requires_grad = unfreeze
    
    def get_rnd_distill_loss_proc(self, distill_loss_acc):
        def proc(h, **kwargs):
            distill_loss = self(h.detach()).mean()
            #print(distill_loss_acc[0].shape, distill_loss.shape)
            distill_loss_acc[0] += distill_loss
            return h
        return proc
        
    def get_rnd_scale_proc(self, distill_loss_acc=None, detach_state_gradient=False):
        def proc(h, **kwargs):
            h_in = h.detach() if detach_state_gradient else h
            distill_loss = self(h_in).unsqueeze(-1)
            g = self.post_scaling_gain * torch.exp(self.scaling_coefficient * distill_loss)
            #print(distill_loss.mean().item(), self.scaling_coefficient.item(), self.post_scaling_gain.item(), g.mean().item())
            if distill_loss_acc is not None:
                distill_loss_acc[0] += distill_loss.mean()
            return h * g
        return proc