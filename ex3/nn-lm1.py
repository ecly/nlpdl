from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

### Exercise
## code adapted from G.Neubig

## 1) Finish the code to implement the feedforward NN Language model as shown in class
#     (i.e., specify the model parameters, the rest is all there). Train the LM.
## 2) Implement an RNN-based LM - nn-lm2 - (you can start from the code basis in nn-lm2).
##    Compare the output the two LMs generate - what do you observe?

N = 2 # The length of the n-gram
EMB_SIZE = 128 # The size of the embedding
HID_SIZE = 128 # The size of the hidden layer

# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already pre-processed.
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(" ")]


# Read in the data
train = list(read_dataset("data/ptb/train_small.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = max(w2i.values()) + 1

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)


### Approach 1: LM as FFNN
# Define the model parameters - YOUR CODE HERE
ntags = len(w2i)
W1 = model.add_parameters((HID_SIZE, EMB_SIZE))
b1 = model.add_parameters(HID_SIZE)
W_sm = model.add_parameters((ntags, HID_SIZE))          # Softmax weights
b_sm = model.add_parameters((ntags))                      # Softmax bias

## end your code here

## Define the connections/graph - YOUR CODE HERE
# A function to calculate scores for one window of words
def calc_score_of_history_ffnn(words):
    # Lookup the embeddings and concatenate them

    # Create the hidden layer
    # hint: you can use affine_transform to calculate Wx+b

    # Calculate the score and return
    dy.renew_cg()
    n_hot = dy.inputVector(words)
    h = dy.tanh(dy.parameter(W1) * n_hot + dy.parameter(b1))
    score = dy.parameter(W_sm) * h + dy.parameter(b_sm)
    return score
## end your code here

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
    # Create a computation graph
    dy.renew_cg()
    # The initial history is equal to end of sentence symbols
    hist = [S] * N
    # Step through the sentence, including the end of sentence token
    all_losses = []
    for next_word in sent + [S]:
        s = calc_score_of_history_ffnn(hist)
        all_losses.append(dy.pickneglogsoftmax(dy.parameter(s), next_word))
        hist = hist[1:] + [next_word] # adds predicted word to history
    return dy.esum(all_losses)

MAX_LEN = 100
# Generate a sentence
def generate_sent():
    dy.renew_cg()
    hist = [S] * N
    sent = []
    while True:
        p = dy.softmax(calc_score_of_history_ffnn(hist)).npvalue()
        next_word = np.random.choice(nwords, p=p/p.sum())
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = hist[1:] + [next_word]
    return sent

print("train..")
for ITER in range(20):
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss.value()
        train_words += len(sent)
        my_loss.backward()
        trainer.update()
        if (sent_id+1) % 1000 == 0:
            print("--finished %r sentences" % (sent_id+1))
            print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_sent_loss(sent)
        dev_loss += my_loss.value()
        dev_words += len(sent)
        trainer.update()
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), time.time()-start))
    # Generate a few sentences
    for _ in range(5):
        sent = generate_sent()
        print(" ".join([i2w[x] for x in sent]))
