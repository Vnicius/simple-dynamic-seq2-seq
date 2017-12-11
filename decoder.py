#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

class Decoder():
    def __init__(self, vocab_size,
                 rnn_size, embeddings):
        
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embeddings = embeddings
        self.decoder_inputs = None
        self.decoder_targets = None
        self.decoder_outputs = None
        self.decoder_state = None
    
    def lstm_model(self, encoder_final_state):
        self.decoder_inputs = tf.placeholder(shape=(None, None),
                                             dtype=tf.int32,
                                             name='decoder_input')
        
        self.decoder_targets = tf.placeholder(shape=(None, None),
                                              dtype=tf.int32,
                                              name='decoder_targets')
        
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        decoder_cell = rnn.LSTMCell(self.rnn_size)

        self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded,
                                                                     initial_state=encoder_final_state,
                                                                     dtype=tf.float32, time_major=True,
                                                                     scope='plain_decoder')
        
        return self.decoder_outputs, self.decoder_state
    
    def optimization(self):
        decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
        decoder_prediction = tf.argmax(decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.decoder_targets,
                                                                                           depth=self.vocab_size,
                                                                                           dtype=tf.float32),
                                                                         logits=decoder_logits)

        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        return train_op, loss, decoder_prediction