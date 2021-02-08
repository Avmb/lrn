###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import os, sys
import time
import math
import pickle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

parser.add_argument('--enable_unk', action='store_true',
                    help='allow generation of <unk> tokens')

# LM robustness
parser.add_argument('--ndistilstudents', type=int, default=0,
                    help='number state  distillation students per layer')
parser.add_argument('--distillossw', type=float, default=1.0,
                    help='student distillation loss weight')
parser.add_argument('--no_average_ensemble', action='store_true',
                    help='disable average ensemble, use only master')

# LM robusness (RND)

parser.add_argument('--rnd_scaling_coefficient', type=float, default=-1.0,
                    help='scaling coefficient in RND') 
parser.add_argument('--rnd_enable', action='store_true',
                    help='enable RND model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

vocab_path = os.path.join(os.path.dirname(args.checkpoint), 'vocab.pickle')
with open(vocab_path, 'rb') as vocab_file:
    vocab = pickle.load(vocab_file) 
ntokens = len(vocab)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.rnd_enable:
    for rnd_model in model.rnd_models:
        rnd_model.scaling_coefficient = torch.scalar_tensor(args.rnd_scaling_coefficient)
        
if args.cuda:
    model.cuda()
    parallel_model = nn.DataParallel(model, dim=1)
else:
    model.cpu()

hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long())
if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    with torch.no_grad():
        for i in range(args.words):
            output, hidden = parallel_model(*hidden, input=input, return_prob=False, 
                                            average_ensemble=not args.no_average_ensemble,
                                            enable_rnd_tune=args.rnd_enable)
            word_weights = output.squeeze().data.div(args.temperature).exp()
            if not args.enable_unk:
                word_weights[0] = 0.0
            word_weights = word_weights.cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = vocab.idx2word[word_idx]
            if word == '<eos>':
                outf.write('\n')
            else:
                outf.write(word + ' ')
            #outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
