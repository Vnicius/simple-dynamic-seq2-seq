#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def random_data(min_len, max_len,
                min_value, max_value,
                batch_size):

    random_size = None

    if min_len > max_len:
        raise ValueError('Minimum length > Maximum length!')
    elif min_len == max_len:
        random_size = lambda min_len, max_len: min_len
    else:
        random_size = np.random.randint
    
    while True:
        yield [np.random.randint(low=min_value, high=max_value,
                                 size=random_size(min_len, max_len + 1)).tolist()
                for _ in range(batch_size)]

if __name__ == "__main__":
    batches = random_data(2, 5, 2, 8, 100)

    for seq in next(batches)[:10]:
        print(seq)