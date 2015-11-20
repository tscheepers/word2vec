import math
import sys
import numpy as np


def train(fi, fo, neg, dim, starting_alpha, win, min_count):

    # Read train file to init vocab
    vocab = Vocabulary(fi, min_count)

    # Init net
    nn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))
    nn1 = np.zeros(shape=(len(vocab), dim))

    global_word_count = 0
    table = TableForNegativeSamples(vocab)

    fi = open(fi, 'r')

    alpha = starting_alpha
    word_count = 0
    last_word_count = 0

    for line in fi:

        tokens = vocab.indices(['{startofline}'] + line.split() + ['{endofline}'])

        for token_idx, token in enumerate(tokens):
            if word_count % 10000 == 0:
                global_word_count += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                sys.stdout.write("\rProgress: %d of %d" % (global_word_count, vocab.word_count))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(token_idx - current_win, 0)
            context_end = min(token_idx + current_win + 1, len(tokens))
            context = tokens[context_start:token_idx] + tokens[token_idx+1:context_end] # Turn into an iterator?

            for context_word in context:
                # Init neu1e with zeros
                neu1e = np.zeros(dim)
                classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                for target, label in classifiers:
                    z = np.dot(nn0[context_word], nn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * nn1[target]              # Error to backpropagate to nn0
                    nn1[target] += g * nn0[context_word] # Update nn1

                # Update nn0
                nn0[context_word] += neu1e

            word_count += 1

    global_word_count += (word_count - last_word_count)
    sys.stdout.write("\rProgress: %d of %d" % (global_word_count, vocab.word_count))
    sys.stdout.flush()
    # fi.close()

    # Save model to file
    save(vocab, nn0, fo)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        # Add special token for start of line and end of line
        for token in ['{startofline}', '{endofline}']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(Word(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(Word(token))
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1

            vocab_items[vocab_hash['{startofline}']].count += 1
            vocab_items[vocab_hash['{endofline}']].count += 1
            word_count += 2

        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Remove rare words and sort
        tmp = []
        tmp.append(Word('{rare}'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['{rare}'] for token in tokens]


class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constants

        table_size = 1e8
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, nn0, fo):
    dim = len(nn0[0])
    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(nn0), dim))
    for token, vector in zip(vocab, nn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

if __name__ == '__main__':

    # Number of negative examples
    negex = 5
    # Dimensionality of word embeddings
    dim = 100
    # Initial learning rate
    alpha = 0.025
    # Max window length
    window = 5
    #Min count for words used to le{rare}unk>
    min_count = 5

    train('text-aa', 'output', negex, dim, alpha, window, min_count)
