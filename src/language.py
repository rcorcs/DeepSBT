# -*- coding: utf-8 -*-
"""
For more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <https://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__


**Requirements**
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import sys
import pickle
import os

import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of several assembly functions
# compiled down to both the x86 and the ARM architectures.
#

######################################################################
# We will be representing each token in a given language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We perform some normalization to trim the data to only use a few
# thousand tokens per language.
#

######################################################################
# We'll need a unique index per word (or token) to use as the inputs
# and targets of the networks later. To keep track of all this we will
# use a helper class called ``Lang`` which has
# word → index (``word2index``) and
# index → word (``index2word``) dictionaries, as well as a count of each
# word ``word2count`` to use to later replace rare words.
#

class Lang:
  SOS_token = 0
  EOS_token = 1

  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "<SOS>", 1: "<EOS>"}
    self.n_words = 2  # Count SOS and EOS

  def addSentence(self, sentence):
    for word in sentence.split():
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1


######################################################################

# Lowercase, trim, and remove non-letter characters

def renameIdsToPlaceholders(s):
  code = []
  start = False
  stop = False
  fName = None
  for line in s.split(';'):
    if len(line.strip())==0:
      continue
    entries = line.split()
    if entries[-1]=='@function':
      start = True
      fName = entries[1]
    if entries[0]=='.cfi_endproc':
      stop = True
    if start:
      code.append(line)
    if stop:
      break
  if len(code)==0:
    return None, None
  s = ' ; '.join(code)
  #s = s.replace(fName,'func_name')
  return fName, s

def isNumeric(s):
  import string
  s = s.strip()
  if len(s)==0:
    return False
  if s[0]=='-':
    s = s[1:]
  if s.startswith('0x'):
    s = s[2:]
    for c in s:
      if c not in string.hexdigits:
        return False
    return True
  else:
    return s.isnumeric()

def isRegister(lang, s):
  if lang=='x86' and s.startswith('%'):
    return True
  if lang=='arm' and len(s)>=2:
    if s[0] in 'xwbhsdqv' and isNumeric(s[1:]):
      return True
    if s in ['sp','pc','xzr','wzr','lr']:
      return True
    #relocation 
    if s in ['abs_g0','abs_g0_nc','abs_g1','abs_g1_nc','abs_g2','abs_g2_nc','abs_g3','abs_g0_s','abs_g1_s','abs_g2_s']:
      return True
    if s in ['pg_hi21','pg_hi21_nc','lo12']:
      return True
  return False

def breakIntegerConstants(lang, s, fName):
  import re
  idpttrn = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
  code = []
  if not s:
    return None
  for line in s.split(';'):
    entries = line.split()
    ne = []
    if entries[0]==fName:
      ne.append('func_name')
    else:
      ne.append(entries[0])
    for e in entries[1:]:
      if isNumeric(e):
        for c in e.strip():
          ne.append(c)
      elif (not isRegister(lang, e)) and idpttrn.match(e):
        if e==fName:
          ne.append('func_name')
        else:
          ne.append('ID')
      else:
        ne.append(e)
    code.append(' '.join(ne))
  if len(code)==0:
    return None
  s = ' ; '.join(code)
  return s
    

def normalizeString(lang, s):
  ns = ''
  for c in s:
    if c in '[()]:;,!-$#':
      ns += ' '+c+' '
    else:
      ns += c
  fName, ns = renameIdsToPlaceholders(ns)
  ns = breakIntegerConstants(lang, ns, fName)
  if ns!=None:
    tokens = ns.split()
    start = tokens.index('.cfi_startproc') + 2
    end = tokens.index('.Lfunc_end0')
    ns = ' '.join(tokens[start:end])
  return ns

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, file1, lang2, file2):
  print("Reading lines...")

  data1 = {}
  with open(file1) as f:
    for line in f:
      line = line.strip()
      if len(line)==0:
        continue
      idx = line.find('content:')
      filename = line[len('file:'):idx].strip()
      code = line[idx+len('content:'):].strip()
      data1[filename] = code
    
  pairs = []
  with open(file2) as f:
    for line in f:
      line = line.strip()
      if len(line)==0:
        continue
      idx = line.find('content:')
      filename = line[len('file:'):idx].strip()
      code = line[idx+len('content:'):].strip()
      if filename in data1.keys():
        code1 = normalizeString(lang1, data1[filename])
        code2 = normalizeString(lang2, code)
        if code1!=None and code2!=None:
          pairs.append( [code1, code2] )

  input_lang = Lang(lang1)
  output_lang = Lang(lang2)

  return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

#MAX_LENGTH = 128

def filterPair(p, max_length):
  return len(p[0].split()) < max_length and \
         len(p[1].split()) < max_length


def filterPairs(pairs, max_length):
  return [pair for pair in pairs if filterPair(pair, max_length)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, file1, lang2, file2, max_length=128):
  input_lang, output_lang, pairs = readLangs(lang1, file1, lang2, file2)
  print("Read %s sentence pairs" % len(pairs))
  pairs = filterPairs(pairs, max_length)
  print("Trimmed to %s sentence pairs" % len(pairs))
  print("Counting words...")
  for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])
  print("Counted words:")
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang, pairs


def loadCachedOrBuild(path, lang1, file1, lang2, file2, max_length=128):
  path_lang1=path+'/'+lang1+'.'+str(max_length)+'.pkl'
  path_lang2=path+'/'+lang2+'.'+str(max_length)+'.pkl'
  path_pairs=path+'/entries.'+str(max_length)+'.pkl'
  input_lang = None
  output_lang = None
  pairs = None
  if os.path.exists(path_lang1) and os.path.exists(path_lang2) and os.path.exists(path_pairs):
    with open(path_lang1, 'rb') as f:
      input_lang = pickle.load(f)
    with open(path_lang2, 'rb') as f:
      output_lang = pickle.load(f)
    with open(path_pairs, 'rb') as f:
      pairs = pickle.load(f)
  else:
    input_lang, output_lang, pairs = prepareData(lang1, file1, lang2, file2, max_length)
    with open(path_lang1, 'wb') as f:
      pickle.dump(input_lang, f)
    with open(path_lang2, 'wb') as f:
      pickle.dump(output_lang, f)
    with open(path_pairs, 'wb') as f:
      pickle.dump(pairs, f)
  return input_lang, output_lang, pairs

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='deepsbt')
  parser.add_argument('--max-length', nargs=1, type=int, default=[128], help='Maximum number of tokens per binary code')
  args = parser.parse_args(sys.argv[1:])

  MAX_LENGTH=int(args.max_length[0])
  PATH = '../data/'

  input_lang, output_lang, pairs = loadCachedOrBuild(PATH, 'x86', PATH+'/x86.txt', 'arm', PATH+'/arm.txt', MAX_LENGTH)
  
  print('Total pairs:',len(pairs))
