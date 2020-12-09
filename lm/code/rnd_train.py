import argparse
import os, sys
import time
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gc

import data
import model

from utils import batchify, get_batch, eval_batchify, get_eval_batch, repackage_hidden, create_exp_dir, save_checkpoint
from weight_drop import WeightDrop
from rnd_model import RND

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=-1,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='Non mono')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=10,
                    help='number of experts')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=False, action='store_true', 
                    help='use single GPU')

# Util
parser.add_argument('--vocab_min_count', type=int, default=5,
                    help='minimum word count when building the vocabulary')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience, set to negative to disable')

# LM robustness (distillation)
parser.add_argument('--ndistilstudents', type=int, default=0,
                    help='number state  distillation students per layer')
parser.add_argument('--distillossw', type=float, default=1.0,
                    help='student distillation loss weight')
parser.add_argument('--unigram_prob_on_zero', action='store_true',
                    help='configure the model such as a zero state results in unigram probability')

# LM robusness (RND)
parser.add_argument('--rnd_n_internal_hid', type=int, default=-1,
                    help='number of internal hidden units per layer in RND')
parser.add_argument('--rnd_n_proj', type=int, default=-1,
                    help='number of projection units per layer in RND')
parser.add_argument('--rnd_n_base_layers', type=int, default=2,
                    help='number of base layers in RND')
parser.add_argument('--rnd_n_student_resnet_blocks', type=int, default=1,
                    help='number of student resnet blocks in RND')
parser.add_argument('--rnd_use_layernorm', action='store_true',
                    help='use layernorm in RND')              
parser.add_argument('--rnd_scaling_coefficient', type=float, default=-1.0,
                    help='scaling coefficient in RND')                    
                    
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size    
if args.rnd_n_internal_hid < 0:
    args.rnd_n_internal_hid = args.nhid
if args.rnd_n_proj < 0:
    args.rnd_n_proj = args.nhid

log_file = os.path.join(args.save, 'rnd_train_log.txt')
print('rnd_train load path: {}/model.pt'.format(args.save))
print('log save path: {}'.format(log_file))
print('model save path: {}/model_with_rnd.pt'.format(args.save))

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_file, 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

vocab_path = os.path.join(args.save, 'vocab.pickle')
with open(vocab_path, 'rb') as vocab_file:
   vocab = pickle.load(vocab_file) 

corpus = data.Corpus(args.data, vocab=vocab, vocab_min_count=args.vocab_min_count)
vocab = corpus.vocab

#eval_batch_size = 10
#test_batch_size = 1
eval_batch_size=args.batch_size
test_batch_size=args.batch_size
train_data = batchify(corpus.train, args.batch_size, args)
val_data, val_mask = eval_batchify(corpus.valid, eval_batch_size, args)
test_data, test_mask = eval_batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.vocab)
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model_with_rnd.pt'))
else:
    model = torch.load(os.path.join(args.save, 'model.pt'))
    # Create one RND Model for each RNN layer
    rnd_models = []
    for l in range(model.nlayers):
        rnn = model.rnns[l]
        if type(rnn) is WeightDrop:
            rnn = rnn.module
        rnd_model = RND(input_dim=rnn.hidden_size, proj_dim=args.rnd_n_proj, 
                        rn_n_hidden_layers=args.rnd_n_base_layers, rn_hidden_dim=args.rnd_n_internal_hid,
                        student_resnet_n_blocks=args.rnd_n_student_resnet_blocks, student_resnet_inner_dim=args.rnd_n_internal_hid,
                        use_layer_norm=args.rnd_use_layernorm, scaling_coefficient=args.rnd_scaling_coefficient)
        rnd_models.append(rnd_model)
    model.rnd_models = nn.ModuleList(rnd_models)
    model.freeze_for_rnd_distillation()
    
if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model

total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, data_source_mask, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets, mask = get_eval_batch(data_source, data_source_mask, i, args)
            masked_targets = targets * mask -100 * (1-mask)     # -100 is the masking value for targets
            masked_targets = masked_targets.view(-1)

            parallel_rv = parallel_model(*hidden, input=data, average_ensemble=True, return_student_distill_loss=True, flatten_returned_lists=True, enable_rnd_distill=True)
            log_prob, student_distill_loss = parallel_rv[0], parallel_rv[-1]
            student_distill_loss = student_distill_loss.sum()
            #loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), masked_targets, ignore_index=-100).data
            loss = student_distill_loss

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.vocab)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.eval() # disable dropout
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            parallel_rv = parallel_model(*hidden[s_id], input=cur_data, return_h=True, return_student_distill_loss=True, flatten_returned_lists=True, enable_rnd_distill=True)
            # reassemble return values
            log_prob, student_distill_loss = parallel_rv[0], parallel_rv[-1].sum()
            parallel_rv = np.array(parallel_rv[1:-1]).reshape((3, -1)).tolist()
            hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_rv
            
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            # Student distillation loss
            #loss = loss + args.distillossw * student_distill_loss
            loss = student_distill_loss
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | distill loss {:5.4f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), 
                student_distill_loss.data * args.small_batch_size / args.batch_size))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, 'rnd_train_optimizer.pt'))
#        if 't0' in optimizer_state['param_groups'][0]:
#            optimizer = torch.optim.ASGD(optimizer_params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
#        else:
#            optimizer = torch.optim.SGD(optimizer_params, lr=args.lr, weight_decay=args.wdecay)
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.wdecay)
        optimizer.load_state_dict(optimizer_state)
    else:
        #optimizer = torch.optim.SGD(optimizer_params, lr=args.lr, weight_decay=args.wdecay)
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.wdecay)

    patience = args.patience
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            #if prm.requires_grad:
            #    prm.data = optimizer.state[prm]['ax'].clone()

        val_loss2 = evaluate(val_data, val_mask, eval_batch_size)
        logging('-' * 89)
        logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss2, math.exp(val_loss2)))
        logging('-' * 89)

        if val_loss2 < stored_loss:
            save_checkpoint(model, optimizer, args.save, rnd_train=True)
            logging('Saving!')
            stored_loss = val_loss2
            patience = args.patience
            
        patience -= 1
        if patience == 0:
            logging('Early stopping!')
            raise KeyboardInterrupt()

        for prm in model.parameters():
            prm.data = tmp[prm].clone()


except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model_with_rnd.pt'))
parallel_model = nn.DataParallel(model, dim=1).cuda()
#parallel_model = model.cuda()

# Run on test data.
test_loss = evaluate(test_data, test_mask, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
