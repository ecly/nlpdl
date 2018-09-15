from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

## Use an RNN instead of a FFNN for language modeling

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
#train = list(read_dataset("data/ptb/train1001.txt"))

w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = max(w2i.values()) + 1 

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)


### Approach 2: RNN as LM

# Define the model (the rnn, input and internal parameters)  - YOUR CODE HERE



## end your code here

def calc_score_of_history_rnn(s, word):

    emb = W_emb[word]
    s = s.add_input(emb)

    # Calculate the score and return
    h = s.output()
    ## your code here


    return 0 


# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
    # Create a computation graph
    dy.renew_cg()

    # Lookup the embeddings and concatenate them
    s = builder.initial_state()

    # Step through the sentence, including the end of sentence token
    all_losses = []
    sentence = [S] + sent + [S]
    for word, next_word in zip(sentence, sentence[1:]):
        score = calc_score_of_history_rnn(s, word)
        all_losses.append(dy.pickneglogsoftmax(score, next_word))
        #hist = hist[1:] + [next_word]
        #hist = hist + [next_word]  - WHY DO WE NO LONGER NEED THIS?
    return dy.esum(all_losses)

MAX_LEN = 100
# Generate a sentence
def generate_sent():
    dy.renew_cg()
    s = builder.initial_state()
    # start the rnn by inputting "<s>"
    hist = S
    sent = []
    while True:
        p = dy.softmax(calc_score_of_history_rnn(s,hist)).npvalue()
        next_word = np.random.choice(nwords, p=p/p.sum())
        #print(next_word, i2w[next_word])
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = next_word
    return sent

print("train..")
for ITER in range(100):
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
