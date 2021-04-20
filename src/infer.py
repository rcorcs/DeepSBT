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

parser = argparse.ArgumentParser(description='deepsbt')
parser.add_argument('--max-length', nargs=1, type=int, default=[128], help='Maximum number of tokens per binary code')
parser.add_argument('--cpu', nargs='?', const=True, default=False, help='Use only CPU for training and inference')
args = parser.parse_args(sys.argv[1:])

MAX_LENGTH=int(args.max_length[0])
#MAX_LENGTH = 128

if args.cpu:
    device = 'cpu'
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = '../data/'

import language
from language import Lang
import model

langs = ('x86', 'arm')
input_lang, output_lang, pairs = language.loadCachedOrBuild(PATH, langs[0], PATH+'/'+langs[0]+'.txt', langs[1], PATH+'/'+langs[1]+'.txt', MAX_LENGTH)

langs = (input_lang, output_lang)

tensor_pairs = [model.tensorsFromPair(p, langs, device) for p in pairs]

hidden_size = 256

ENCODER_PATH=PATH+'/encoder.'+str(MAX_LENGTH)+'.'+str(hidden_size)+'.pt'
DECODER_PATH=PATH+'/decoder.'+str(MAX_LENGTH)+'.'+str(hidden_size)+'.pt'

if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH):
  encoder1 = torch.load(ENCODER_PATH)
  attn_decoder1 = torch.load(DECODER_PATH)
else:
  encoder1 = model.EncoderRNN(input_lang, hidden_size).to(device)
  #attn_decoder1 = AttnDecoderRNN(output_lang, hidden_size, dropout_p=0.1).to(device)
  attn_decoder1 = model.DecoderRNN(output_lang, hidden_size).to(device)

input_code = language.normalizeString(input_lang, '.text;.globl swap;.p2align 4, 0x90;.type swap,@function;swap:;.cfi_startproc;movl (%rdi), %eax;movl (%rsi), %ecx;movl %ecx, (%rdi);movl %eax, (%rsi);retq;.Lfunc_end0:;.size swap, .Lfunc_end0-swap;.cfi_endproc;.section ".note.GNU-stack","",@progbits;.addrsig;')


input_tensor = model.tensorFromSentence(input_lang, input_code, device)
output_tensor, _ = model.evaluate(encoder1, attn_decoder1, input_tensor, MAX_LENGTH, device)

output_code = model.sentenceFromTensor(output_lang, output_tensor)
print(output_code)

