#! /usr/bin/env python3

import sys

import data

unk_token='<unk>'
file_names=['train.txt', 'valid.txt', 'test.txt']

def main():
  if (len(sys.argv) != 4):
    usage()
  train_set_dir, in_dir, out_dir = sys.argv[1:]

  train_corpus = data.Corpus(train_set_dir)
  train_words = train_corpus.dictionary.word2idx.keys()

  for file_name in file_names:
    with open(in_dir+'/'+file_name) as in_fs, open(out_dir+'/'+file_name, 'w') as out_fs:
      for line in in_fs:
        tokens = line.strip().lower().split()
        re_tokens = [token if (token in train_words) else unk_token for token in tokens]
        print(" ".join(re_tokens), file=out_fs)

def usage():
  print("Usage:", file=sys.stderr)
  print(sys.argv[0], "train_set_dir in_dir out_dir", file=sys.stderr)
  sys.exit(-1)

if __name__ == '__main__':
  main()

