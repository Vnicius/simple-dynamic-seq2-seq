#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

class Encoder():

    def __init__(self, vocab_size,
                 rnn_size, embeddings):
        
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embeddings = embeddings
        self.encoder_inputs = None
        self.encoder_outputs = None
        self.encoder_final_state = None

    def get_state(self):
        self.encoder_inputs = tf.placeholder(shape=(None, None),
                                             dtype=tf.int32,
                                             name='encoder_inputs')
        
        encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        encoder_cell =  rnn.LSTMCell(self.rnn_size)

        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                                           dtype=tf.float32, time_major=True)
        
        return self.encoder_final_state