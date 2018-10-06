import sys
from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

"""
Paper optimizes:
p(label | sent, position of entities)

We want:
p(label | e1, e2)
1) FFNN
2) LSTM
"""


def get_vocabulary(filename):
    w2i = defaultdict(lambda: 0)
    l2i = defaultdict(lambda: 0)
    with open(filename, "r") as f:
        for line in f.readlines():
            _, relation_cls, _, _, sentence = line.split("\t")
            for word in sentence.strip().split(" "):
                if word not in w2i:
                    w2i[word] = len(w2i)
                if relation_cls not in l2i:
                    l2i[relation_cls] = len(l2i)

    return w2i, l2i


def load_instances(w2i, l2i, filename):
    """ Takes word2id and label2id and filename to produce instances """
    with open(filename, "r") as f:
        for line in f.readlines():
            _, relation_cls, idx1, idx2, sentence = line.split("\t")
            embedding = [w2i[word] for word in sentence.strip().split(" ")]
            yield (int(idx1), int(idx2), embedding, l2i[relation_cls])


def convert_to_n_hot(X, vocab_size):
    out = []
    for idx1, idx2, words, label in X:
        n_hot = np.zeros(vocab_size)
        n_hot[words[idx1]] = 1
        n_hot[words[idx2]] = 1
        out.append((n_hot, label))

    return out


def prepare_data(train_file, dev_file):
    train_file = "train.txt"
    dev_file = "test.txt"
    w2i, l2i = get_vocabulary(train_file)
    vocab_size = max(w2i.values()) + 1
    ntags = len(l2i)

    i2w = {v: k for k, v in w2i.items()}
    i2l = {v: k for k, v in l2i.items()}
    train = list(load_instances(w2i, l2i, train_file))
    train = convert_to_n_hot(train, vocab_size)
    dev = list(load_instances(w2i, l2i, dev_file))
    dev = convert_to_n_hot(dev, vocab_size)

    random.shuffle(train)
    random.shuffle(dev)

    return train, dev, w2i, l2i, vocab_size, ntags


    return model, trainer


train, dev, w2i, l2i, vocab_size, ntags = prepare_data("train.txt", "test.txt")
hidden_size = 100

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)
W1 = model.add_parameters((hidden_size, vocab_size))
b1 = model.add_parameters(hidden_size)
W_sm = model.add_parameters((ntags, hidden_size))  # Softmax weights
b_sm = model.add_parameters((ntags))  # Softmax bias

# A function to calculate scores for one value
def calc_scores(input_vec):
    dy.renew_cg()
    n_hot = dy.inputVector(input_vec)
    h = dy.tanh(dy.parameter(W1) * n_hot + dy.parameter(b1))
    score = dy.parameter(W_sm) * dy.rectify(h) + dy.parameter(b_sm)
    return score


for iteration in range(20):
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for x, y in train:
        my_loss = dy.pickneglogsoftmax(calc_scores(x), y)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
    print(
        "iter %r: train loss/sent=%.4f, time=%.2fs"
        % (iteration, train_loss / len(train), time.time() - start)
    )
    # Perform testing
    test_correct = 0.0
    for x, y in dev:
        predict = dy.softmax(calc_scores(x)).npvalue().argmax()
        if predict == y:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (iteration, test_correct / len(dev)))


