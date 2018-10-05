from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

"""
Paper optimizes:
Optimize p(label | sent, position of entities)

We want:
p(label | e1, e2)
    1) FFNN
    2) LSTM

p(label | sent, e1, e2)
"""

def get_vocabularily(datafile):
    """
    output 4 dictionaries
    map from word -> ints/ids
    map from int/ids -> word
    map from label -> ints/ids
    map from int/ids -> label
    """
    return None

def load_instances(word2id, label2id):
    """
    Takes word2id and label2id and filename
    """
    return None
