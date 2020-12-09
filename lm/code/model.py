import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import rnn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers, 
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, 
                 tie_weights=False, ldropout=0.5, n_experts=10, ndistilstudents=0, 
                 unigram_prob_on_zero=False, unigram_frequencies=None,
                 rnd_models=None):
        super(RNNModel, self).__init__()
        self.ndistilstudents=ndistilstudents
        self.unigram_prob_on_zero=unigram_prob_on_zero
        self.unigram_frequencies=unigram_frequencies
        self.rnd_models=rnd_models
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        
        rnn_type = rnn_type.lower()
        self.rnns = [rnn.RNN(rnn_type, ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0, n_students=ndistilstudents) for l in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['_W', '_U'], dropout=wdrop if self.use_dropout else 0) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        
        self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        latent_linear = nn.Linear(nhidlast, n_experts*ninp, bias=not unigram_prob_on_zero)
        #latent_linear = nn.Linear(nhidlast, n_experts*ninp, bias=True)
        self.latent = nn.Sequential(latent_linear, nn.Tanh())
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.bias.data[:] = torch.zeros_like(self.decoder.bias.data)
        self.decoder_gain = nn.Parameter(torch.ones(ninp), requires_grad=False)
        #self.decoder_gain = nn.Parameter(torch.scalar_tensor(1.0), requires_grad=True)
        #self.decoder_gain = nn.Parameter(torch.zeros(ninp))

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        if unigram_prob_on_zero:
            self.decoder.bias.requires_grad=False
            pass
            
        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.unigram_prob_on_zero:
            log_probs = torch.log(self.unigram_frequencies)
            #log_probs = log_probs - log_probs.max()
            log_probs_safe = log_probs[log_probs > -float("Inf")]
            log_probs = log_probs - log_probs_safe.mean()
            log_probs.masked_fill_(log_probs <= -float("Inf"), -1e18)
            self.decoder.bias.data[:] = log_probs
            self.decoder_gain.data = self.decoder_gain.data * log_probs_safe.std()
            #print(self.decoder.bias)
            #print(self.decoder_gain)
            pass
            

    def forward(self, *hidden, input=None, return_h=False, return_prob=False, 
                return_student_distill_loss=False, average_ensemble=False, 
                enable_rnd_distill=False, enable_rnd_tune=False,
                flatten_returned_lists=False):
        batch_size = input.size(1)

        if self.rnn_type == "lstm" or self.rnn_type == "sru":
            # hidden state must be rearranged a (h, c) tuple
            rearranged_hidden = []
            for i in range(0, len(hidden), 2):
                rearranged_hidden.append((hidden[i], hidden[i+1]))
            hidden = rearranged_hidden
        
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if (self.training and self.use_dropout) else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        distill_loss_acc = [torch.tensor(0.0).to(input.device)] if return_student_distill_loss else None
        for l, rnn in enumerate(self.rnns):
            state_post_proc = None
            assert(not (enable_rnd_distill and enable_rnd_tune)), "enable_rnd_distill and enable_rnd_tune can't be enabled at the same time"
            if enable_rnd_distill:
                state_post_proc = self.rnd_models[l].get_rnd_distill_loss_proc(distill_loss_acc)
            if enable_rnd_tune:
                state_post_proc = self.rnd_models[l].get_rnd_scale_proc(distill_loss_acc)

            current_input = raw_output
            if self.ndistilstudents  == 0:
                raw_output, new_h = rnn(current_input, hidden[l], 
                                        distill_loss_acc=distill_loss_acc, state_post_proc=state_post_proc)
            else:
                raw_output, new_h = rnn(current_input, hidden[l], distill_loss_acc=distill_loss_acc, average_ensemble=average_ensemble, 
                                        state_post_proc=state_post_proc)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl if self.use_dropout else 0)
        logit = self.decoder(latent.view(-1, self.ninp) * self.decoder_gain)
        #print(self.decoder_gain.max().item(), self.decoder_gain.min().item(), self.decoder_gain.mean().item())

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit, -1)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken), -1).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        rv = (model_output, hidden)
        if return_h:
            rv = rv + (raw_outputs, outputs)
        if return_student_distill_loss:
            rv = rv + (distill_loss_acc[0].reshape([1, 1]), )
        if flatten_returned_lists:
            new_rv = []
            for e in rv:
                if isinstance(e, list):
                    for ee in e:
                        new_rv.append(ee)
                else:
                    new_rv.append(e)
            rv = new_rv
        return rv

    def init_hidden(self, bsz):
        #print(self.rnn_type)
        weight = next(self.parameters()).data
        if self.rnn_type == "lstm" or self.rnn_type == "sru":
            #return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
            #         Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
            #        for l in range(self.nlayers)]
            h_acc = []
            for l in range(self.nlayers):
                h_acc.append(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                h_acc.append(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
            return h_acc
        else:
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_())
                    for l in range(self.nlayers)]
                    
    def freeze_for_rnd_distillation(self, unfreeze=False):
        rnd_params = {} if self.rnd_models is None else dict(self.rnd_models.named_parameters())
        for param_name, param in self.named_parameters():
            base_param_name = '.'.join(param_name.split('.')[1:])
            if base_param_name not in rnd_params:
                param.requires_grad = unfreeze


if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    hidden = model.init_hidden(9)
    model(hidden, input=input)

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())
