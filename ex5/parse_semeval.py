import re
import nltk
from nltk import word_tokenize
from random import shuffle

TRAIN_FILE = "data/training/TRAIN_FILE.TXT"
PATTERN = r'(\d+)\s"(.*<e1>(.*)<\/e1>.*<e2>(.*)<\/e2>.*)"'
TEST_SPLIT = 0.1

def parse_instance(instance):
    raw, rel = instance.split('\n')[:2]
    rel_class = re.sub(r'\(.*\)', "", rel)
    match = re.match(PATTERN, raw)

    sent_id = match.group(1).strip()
    sent = match.group(2).strip()
    rel1 = match.group(3).strip()
    rel2 = match.group(4).strip()

    new_rel1 = rel1.replace(" ", "_")
    new_rel2 = rel2.replace(" ", "_")

    if re.match(".*(e2,e1)", rel):
        new_rel1, new_rel2 = new_rel2, new_rel1

    sent = sent.replace(rel1, new_rel1)
    sent = sent.replace(rel2, new_rel2)
    trimmed = re.sub(r'<(\/)?e\d>', '', sent)
    tokens = word_tokenize(trimmed)
    if new_rel1 not in tokens or new_rel2 not in tokens:
        #this handles doves<e2>moles</e2> corner case for now
        return []

    idx1 = str(tokens.index(new_rel1))
    idx2 = str(tokens.index(new_rel2))
    return [sent_id, rel_class, idx1, idx2, trimmed]


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

if __name__ == '__main__':
    nltk.download('punkt')
    with open(TRAIN_FILE) as f:
        INSTANCES = [parse_instance(i.strip()) for i in f.read().strip().split('\n\n')]
        output_instances(INSTANCES)
