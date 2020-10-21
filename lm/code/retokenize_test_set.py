#! /usr/bin/env python3

import sys

import data

unk_token=''

def main():
  if (len(sys.argv) != 2):
    usage()
  train_set_dir = sys.argv[1]

  train_corpus = data.Corpus(train_set_dir)
  train_words = train_corpus.dictionary.word2idx.keys()

  for line in sys.stdin:
    tokens = line.strip().split()
    re_tokens = [token if (token in train_words) else unk_token for token in tokens]
    print(" ".join(re_tokens))

def usage():
  print("Usage:", file=sys.stderr)
  print(sys.argv[0], "train_set_dir", file=sys.stderr)
  sys.exit(-1)

if __name__ == '__main__':
  main()

