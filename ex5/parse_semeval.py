import re
import nltk
from nltk import word_tokenize
from random import shuffle

TRAIN_FILE = "data/training/TRAIN_FILE.TXT"
PATTERN = r'(\d+)\s"(.*<e1>(.*)<\/e1>.*<e2>(.*)<\/e2>.*)"'
TEST_SPLIT = 0.1

def parse_tokens(tokens):
    # retrieve indices of relations subtracting the
    # token count we remove from filtered prior to their position
    idx1 = tokens.index("e1") - 1
    idx2 = tokens.index("e2") - 7
    filtered = [x for x in tokens if not re.match(r'<|(/?e(1|2))|>', x)]

    return " ".join(filtered), idx1, idx2


def parse_instance(instance):
    raw, relation = instance.split('\n')[:2]
    relation_cls = re.sub(r'\(.*\)', "", relation)
    match = re.match(PATTERN, raw)
    sentence_idx = match.group(1).strip()
    sentence = match.group(2).strip()
    filtered, idx1, idx2 = parse_tokens(word_tokenize(sentence))
    if "(e2,e1)" in relation:
        idx1, idx2 = idx2, idx1

    return [sentence_idx, relation_cls, str(idx1), str(idx2), filtered]


def print_instances_to_file(inst, filename):
    """ Print instances to a given file separated by tabs """
    with open(filename, 'a') as out_file:
        for i in inst:
            out_file.write('\t'.join(i) + '\n')


def output_instances(inst):
    """ Splits instances into train/test and prints to files """
    shuffle(inst)
    test_idx = int(len(inst) * TEST_SPLIT)
    test = inst[:test_idx]
    train = inst[test_idx:]
    print_instances_to_file(test, "test.txt")
    print_instances_to_file(train, "train.txt")


def main():
    nltk.download('punkt')
    with open(TRAIN_FILE) as f:
        instances = [parse_instance(i.strip()) for i in f.read().strip().split('\n\n')]
        output_instances(instances)

if __name__ == '__main__':
    main()
