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

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--test_data', type=str, default='./penn/',
                    help='location of the test data corpus')
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
parser.add_argument('--lr', type=float, default=30,
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
                    help='random seed')
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

# LM robustness
parser.add_argument('--ndistilstudents', type=int, default=0,
                    help='number state  distillation students per layer')
parser.add_argument('--distillossw', type=float, default=1.0,
                    help='student distillation loss weight')
parser.add_argument('--no_average_ensemble', action='store_true',
                    help='disable average ensemble, use only master')

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
parser.add_argument('--rnd_nofreeze_student', action='store_true',
                    help='freeze the RND student during tuning (ony tunes the post scaling gain parameter)')     

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
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

corpus = data.SentTestCorpus(args.test_data, vocab)

#test_batch_size = 1
test_batch_size = args.batch_size

###############################################################################
# Evaluating code
###############################################################################

def evaluate_by_sentence(test_sentences, test_batch_size, args, average_ensemble):
    for sent_id, sent in enumerate(test_sentences):
        if len(sent) < 2:
            print("smallsent")
            continue
        test_data, test_mask = eval_batchify(sent, test_batch_size, args)
        test_loss, test_student_loss = evaluate(test_data, test_mask, test_batch_size, average_ensemble=average_ensemble)
        print(len(sent), test_loss, math.exp(test_loss), test_student_loss)

def evaluate(data_source, data_source_mask, batch_size=10, average_ensemble=True):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_student_loss = 0
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        data, targets, mask = get_eval_batch(data_source, data_source_mask, 0, args)
        masked_targets = targets * mask -100 * (1-mask)     # -100 is the masking value for targets
        masked_targets = masked_targets.view(-1)

        parallel_rv = parallel_model(*hidden, input=data, average_ensemble=average_ensemble, return_student_distill_loss=True, flatten_returned_lists=True, enable_rnd_tune=True)
        log_prob, student_distill_loss = parallel_rv[0], parallel_rv[-1]
        student_distill_loss = student_distill_loss.mean()
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), masked_targets, ignore_index=-100).data

        total_loss += loss * len(data)
        total_student_loss += student_distill_loss.data * len(data)

        hidden = parallel_rv[1:-1]
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source), total_student_loss.item() / len(data_source)


# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model_with_rnd_tune.pt'))
for rnd_model in model.rnd_models:
    rnd_model.scaling_coefficient = torch.scalar_tensor(args.rnd_scaling_coefficient)
#parallel_model = nn.DataParallel(model.cuda(), dim=1)
parallel_model = model.cuda()

# Run on test data.
evaluate_by_sentence(corpus.test_sentences, test_batch_size, args, average_ensemble=not args.no_average_ensemble)

#logging('=' * 89)
#logging('| Test set: %s' % args.test_data)
#logging('| Evaluation results | test loss {:5.2f} | test ppl {:8.2f} | test distillation loss {:5.4f}'.format(
#    test_loss, math.exp(test_loss), test_student_loss))
#logging('=' * 89)

