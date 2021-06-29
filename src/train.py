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

######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

#teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, teacher_forcing_ratio = 0):
  encoder_hidden = encoder.initHidden(device)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_tensor.size(0)
  target_length = target_tensor.size(0)

  encoder_outputs = None
  if decoder.uses_attention:
    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
      input_tensor[ei], encoder_hidden)
    if encoder_outputs!=None:
      encoder_outputs[ei] = encoder_output[0, 0]

  decoder_input = torch.tensor([[decoder.SOS_token]], device=device)

  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      # Using attention model
      if decoder.uses_attention:
        decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_outputs)
      else:
        decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden)
      loss += criterion(decoder_output, target_tensor[di])
      decoder_input = target_tensor[di]  # Teacher forcing

  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      # Using attention model
      if decoder.uses_attention:
        decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_outputs)
      else:
        decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input

      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == decoder.EOS_token:
          break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, tensor_pairs, n_iters, device, learning_rate=0.01, print_every=1000):
  start = time.time()
  print_loss_total = 0  # Reset every print_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  #training_indices = [ random.randrange(0,len(tensor_pairs)) for i in range(n_iters) ]
  
  criterion = nn.NLLLoss()

  total_iters = n_iters*len(tensor_pairs)
  iter = 0
  for i in range(n_iters):
    for training_pair in tensor_pairs:
      iter += 1
      #for iter in range(1, n_iters + 1):
      #idx = training_indices[iter - 1]
      #training_pair = tensor_pairs[idx]

      input_tensor = training_pair[0]
      target_tensor = training_pair[1]

      loss = train(input_tensor, target_tensor, encoder,
                   decoder, encoder_optimizer, decoder_optimizer, criterion, device)
      print_loss_total += loss

      if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, iter / total_iters),
                                     iter, iter / total_iters * 100, print_loss_avg))
        #check point
        if encoder.path!=None:
          torch.save(encoder, encoder.path)
        if decoder.path!=None:
          torch.save(decoder, decoder.path)
        import psutil
        print('Used memory:',psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent>90:
          #breaks training if memory usage becomes unmanagable
          break


if __name__ == '__main__':

  import sys
  import pickle
  import os

  import argparse

  import language
  from language import Lang

  import model

  parser = argparse.ArgumentParser(description='deepsbt')
  parser.add_argument('--max-length', nargs=1, type=int, default=[128], help='Maximum number of tokens per binary code')
  parser.add_argument('--cpu', nargs='?', const=True, default=False, help='Use only CPU for training and inference')
  args = parser.parse_args(sys.argv[1:])
  
  if args.cpu:
    device = 'cpu'
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  MAX_LENGTH=int(args.max_length[0])
  #MAX_LENGTH = 128
  
  LangPath = '../data/'
  PATH = '../models/'

  langs = ('x86', 'arm')
  input_lang, output_lang, pairs = language.loadCachedOrBuild(PATH, langs[0], LangPath+'/'+langs[0]+'.txt', langs[1], LangPath+'/'+langs[1]+'.txt', MAX_LENGTH)

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
    attn_decoder1 = model.AttnDecoderRNN(output_lang, hidden_size, MAX_LENGTH, dropout_p=0.1).to(device)
    #attn_decoder1 = model.DecoderRNN(output_lang, hidden_size).to(device)

  encoder1.path = ENCODER_PATH
  attn_decoder1.path = DECODER_PATH

  trainIters(encoder1, attn_decoder1, tensor_pairs, 4, device=device, print_every=500)

  torch.save(encoder1, encoder1.path)
  torch.save(attn_decoder1, attn_decoder1.path)


