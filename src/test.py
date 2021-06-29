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

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

if __name__ == '__main__':

  import sys
  import pickle
  import os

  import argparse

  import language
  from language import Lang

  import model

  parser = argparse.ArgumentParser(description='deepsbt')
  parser.add_argument("input", type=str, help="Input assembly source file")
  parser.add_argument('--max-length', nargs=1, type=int, default=[128], help='Maximum number of tokens per binary code')
  parser.add_argument('--cpu', nargs='?', const=True, default=False, help='Use only CPU for training and inference')
  parser.add_argument('--data-path', type=str, default="../data/", help='Maximum number of tokens per binary code')
  parser.add_argument('--models-path', type=str, default="../models/", help='Maximum number of tokens per binary code')
  args = parser.parse_args(sys.argv[1:])
  
  if args.cpu:
    device = 'cpu'
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  MAX_LENGTH=int(args.max_length[0])
  #MAX_LENGTH = 128

  LangPath = args.data_path
  PATH = args.models_path

  langs = ('x86', 'arm')
  input_lang, output_lang, pairs = language.loadCachedOrBuild(PATH, langs[0], LangPath+'/'+langs[0]+'.txt', langs[1], LangPath+'/'+langs[1]+'.txt', MAX_LENGTH)

  langs = (input_lang, output_lang)
  
  tensor_pairs = [model.tensorsFromPair(p, langs, device) for p in pairs]

  hidden_size = 256

  ENCODER_PATH=PATH+'/encoder.'+str(MAX_LENGTH)+'.'+str(hidden_size)+'.pt'
  DECODER_PATH=PATH+'/decoder.'+str(MAX_LENGTH)+'.'+str(hidden_size)+'.pt'

  encoder1 = torch.load(ENCODER_PATH)
  attn_decoder1 = torch.load(DECODER_PATH)

  encoder1.path = ENCODER_PATH
  attn_decoder1.path = DECODER_PATH
  
  inputfile = args.input
  input_code = language.parseRawFile(inputfile)
  input_code, fname, ids = language.normalizeString(input_lang, input_code)
  print(input_code)
  
  input_tensor = model.tensorFromSentence(input_lang, input_code, device)
  output_words, attentions = model.evaluate(encoder1, attn_decoder1, input_tensor, MAX_LENGTH, device)
  output = model.sentenceFromTensor(output_lang, output_words)
  print(output)

