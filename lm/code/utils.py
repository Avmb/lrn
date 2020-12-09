import os, shutil
import numpy as np
import torch
from torch.autograd import Variable

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #print(data.size())
    if args.cuda:
        data = data.cuda()
    return data

def eval_batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts with padding
    nbatch = int(np.ceil(data.size(0) / float(bsz)))
    # Pad
    batches = torch.zeros(nbatch * bsz, dtype=torch.int64)
    batches[:data.size(0)] = data[:]
    # Evenly divide the data across the bsz batches.
    batches = batches.view(bsz, -1).t().contiguous()
    # Create mask
    mask = torch.zeros(nbatch * bsz, dtype=torch.int64)
    mask[:data.size(0)] = torch.ones_like(data, dtype=torch.int64)
    mask = mask.view(bsz, -1).t().contiguous()
    #print(batches.size(), mask.size())
    if args.cuda:
        batches = batches.cuda()
        mask = mask.cuda()
    return batches, mask

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = Variable(source[i+1:i+1+seq_len])
    return data, target
    
def get_eval_batch(source, source_mask, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = Variable(source[i+1:i+1+seq_len])
    mask = Variable(source_mask[i+1:i+1+seq_len])
    return data, target, mask

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, optimizer, path, finetune=False, rnd_train=False, rnd_tune=False):
    if rnd_tune:
        torch.save(model, os.path.join(path, 'model_with_rnd_tune.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'rnd_train_optimizer_tune.pt'))
    elif rnd_train:
        torch.save(model, os.path.join(path, 'model_with_rnd.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'rnd_train_optimizer.pt'))        
    elif finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
