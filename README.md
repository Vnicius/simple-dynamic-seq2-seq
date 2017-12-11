# simple-dynamic-seq2seq
A simple dynamic Sequence-to-sequence with TensorFlow.

*Based in [this](https://github.com/ematvey/tensorflow-seq2seq-tutorials) tutorial*

## Dependences

- Python = 3.x
- TensorFlow = 1.4
- Numpy = 1.13.3

## Run

```console
    $ python3 seq2seq.py [1] [2] [3]
```

1. Number of recurrences
2. Number of epochs of train
3. The size of batch to train

## Examples

**Running:**
```console
    $ python3 seq2seq.py 20 3 1000
```

**Expected output**
```console
batch 0
        minibatch loss: 2.3116953372955322
        sample 1:
                input     > [5 6 5 7 0 0 0 0]
                predicted > [1 1 1 8 0 0 8 7 7]
        sample 2:
                input     > [6 4 3 3 0 0 0 0]
                predicted > [1 1 1 0 8 8 8 8 7]
        sample 3:
                input     > [5 3 5 6 4 3 7 0]
                predicted > [5 8 8 8 8 0 0 0 0]

batch 1000
        minibatch loss: 0.15936240553855896
        sample 1:
                input     > [6 4 3 4 3 0 0 0]
                predicted > [6 4 3 4 3 1 0 0 0]
        sample 2:
                input     > [3 4 2 0 0 0 0 0]
                predicted > [3 4 2 1 0 0 0 0 0]
        sample 3:
                input     > [7 7 6 3 3 4 0 0]
                predicted > [7 7 6 3 3 4 1 0 0]

batch 2000
        minibatch loss: 0.07801636308431625
        sample 1:
                input     > [6 7 7 2 4 0 0 0]
                predicted > [6 7 7 2 4 1 0 0 0]
        sample 2:
                input     > [2 3 5 3 2 5 0 0]
                predicted > [2 3 5 3 2 5 1 0 0]
        sample 3:
                input     > [3 5 6 0 0 0 0 0]
                predicted > [3 5 6 1 0 0 0 0 0]
```
