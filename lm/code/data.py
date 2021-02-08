import os
import torch

from collections import Counter

import pickle

class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        self.add_word('<unk>')
        self.add_word('<eos>')
        self.counter[0] = 0
        self.counter[1] = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        #self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
        
    def get_frequencies(self, device='cpu'):
        counts = [self.counter[word_id] for word_id in range(len(self.idx2word))]
        frequencies = torch.FloatTensor(counts).to(device=device) / float(self.total)
        return frequencies
        
    def build_vocab(self, path, min_count=1):
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.add_word(word)
        
        # Fix counts for <unk> and <eos>
        self.counter[0] = max(self.counter[0], min_count)
        self.counter[1] = max(self.counter[1], min_count)
        self.total = sum(list(self.counter.values()))
        if min_count == 1:
            return
        # prune by word counts
        #new_word2idx = {'<unk>' : 0, '<eos>' : 1}
        #new_idx2word = ['<unk>', '<eos>']
        new_counter = Counter()
        #new_counter[0] = counter[0]
        #new_counter[1] = counter[1]
        new_word2idx = {}
        new_idx2word =  []
        new_total = 0
        for word_id in range(0, len(self.idx2word)): 
            cur_count = self.counter[word_id]
            if cur_count >= min_count:
                word = self.idx2word[word_id]
                new_idx2word.append(word)
                new_id = len(new_idx2word) - 1
                new_word2idx[word] = new_id
                new_counter[new_id] += cur_count
                new_total += cur_count
        self.word2idx = new_word2idx
        self.idx2word = new_idx2word
        self.counter = new_counter
        self.total = new_total

    def tokenize(self, path, device='cpu'):
        """Tokenizes a text file."""
        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.word2idx.get(word, 0)) # 0 is the id of the <unk> word
        ids = torch.LongTensor(ids).to(device=device)
        return ids

    def tokenize_by_sentence(self, path, device='cpu'):
        """Tokenizes a text file."""
        # Tokenize file content
        sent_acc = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                #if len(line.strip()) == 0:
                #    continue
                ids = []
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.word2idx.get(word, 0)) # 0 is the id of the <unk> word
                ids = torch.LongTensor(ids).to(device=device)
                sent_acc.append(ids)
        return sent_acc


class Corpus(object):
    def __init__(self, path, vocab=None, vocab_min_count=1):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocab()
            self.vocab.build_vocab(os.path.join(path, 'train.txt'), min_count = vocab_min_count)
        
        self.train = self.vocab.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.vocab.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.vocab.tokenize(os.path.join(path, 'test.txt'))

class TestCorpus(object):
    def __init__(self, path, vocab):
        self.vocab = vocab
        self.test = self.vocab.tokenize(path)

class SentTestCorpus(object):
    def __init__(self, path, vocab):
        self.vocab = vocab
        self.test_sentences = self.vocab.tokenize_by_sentence(path)
