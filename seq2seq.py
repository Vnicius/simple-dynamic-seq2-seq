#!/usr/bin/python3
# -*- coding:utf-8

'''
    Author: Vinícius Matheus
    GitHub: Vnicius

    A simple version of a dynamic sequece-to-senquence algorithm.
    Based in this tutotial: https://github.com/ematvey/tensorflow-seq2seq-tutorials
'''
import tensorflow as tf
import batch_helper as bh
from encoder import Encoder
from decoder import Decoder

PAD = 0 # full a sequence
EOS = 1 # end of the sequence
VOCAB_SIZE = 10 # 0-9

def embedding(vocab_size, input_embedding_size, dtype=tf.float32):
    return tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=dtype)

def next_feed(batches, encoder_inputs,
              decoder_inputs, decoder_targets):

    batch = next(batches)
    encoder_inputs_, _ = bh.batch(batch)
    decoder_targets_, _ = bh.batch([(sequence) + [EOS] for sequence in batch])
    decoder_inputs_, _ = bh.batch([[EOS] + (sequence) for sequence in batch])

    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

if __name__ == "__main__":
    import sys

    rnn_size = int(sys.argv[1])
    num_epoch = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    embeddings = embedding(VOCAB_SIZE, rnn_size)

    encoder = Encoder(VOCAB_SIZE, rnn_size, embeddings)
    encoder_final_state = encoder.get_state()

    decoder = Decoder(VOCAB_SIZE, rnn_size, embeddings)
    decoder.lstm_model(encoder_final_state)
    train_op, loss, decoder_prediction = decoder.optimization()

    max_batch = ((num_epoch - 1)  * batch_size) + 1

    batches = bh.random_data(3, 8, 2, 8, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for batch in range(max_batch):
            feed = next_feed(batches, encoder.encoder_inputs,
                             decoder.decoder_inputs, decoder.decoder_targets)
            _, l = sess.run([train_op, loss], feed)

            if batch == 0 or batch % batch_size == 0:
                print('batch {}'.format(batch))
                print('\tminibatch loss: {}'.format(sess.run(loss, feed)))
                predict_ = sess.run(decoder_prediction, feed)
                for i, (inp, pred) in enumerate(zip(feed[encoder.encoder_inputs].T, predict_.T)):
                    print('\tsample {}:'.format(i + 1))
                    print('\t\tinput     > {}'.format(inp))
                    print('\t\tpredicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()