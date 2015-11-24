import math
import sys
import numpy as np


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Ngram:
    def __init__(self, ngrams):
        self.ngrams = ngrams
        self.count = 0


class Corpus:
    def __init__(self, filename):
        i = 0
        file_pointer = open(filename, 'r')

        all_tokens = []

        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                all_tokens.append(token)

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpus: %d" % i)

        sys.stdout.flush()
        print "\rCorpus read: %d" % i

        file_pointer.close()

        self.tokens = all_tokens

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Vocabulary:
    def __init__(self, corpus, min_count):
        self.words = []
        self.word_map = {}
        self.buildWords(corpus, min_count)

        self.ngrams = []
        self.ngram_map = {}
        self.buildNgrams(corpus)

    def buildWords(self, corpus, min_count):
        words = []
        word_map = {}

        i = 0
        for token in corpus:
            if token not in word_map:
                word_map[token] = len(words)
                words.append(Word(token))
            words[word_map[token]].count += 1

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rBuilding vocabulary: %d" % i)

        sys.stdout.flush()
        print "\rVocabulary built: %d" % i

        self.words = words
        self.word_map = word_map # Mapping from each token to its index in vocab

        # Remove rare words and sort
        tmp = []
        tmp.append(Word('{rare}'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update word_map
        word_map = {}
        for i, token in enumerate(tmp):
            word_map[token.word] = i

        self.words = tmp
        self.word_map = word_map

    def buildNgrams(self, corpus):

        ngrams = []
        ngram_map = {}

        # Build ngram map
        for n in range(2, 4):
            i = 0
            ngram_l = []
            for token in corpus:
                if len(ngram_l) < n:
                    ngram_l.append(token)
                    continue

                if len(ngram_l) == n:
                    ngram_l.pop(0)

                ngram_l.append(token)
                ngram_t = tuple(ngram_l)

                if ngram_t not in ngram_map:
                    ngram_map[ngram_t] = len(ngrams)
                    ngrams.append(Ngram(ngram_t))

                ngrams[ngram_map[ngram_t]].count += 1

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rBuilding %d-grams: %d" % (n, i))

            sys.stdout.flush()
            print "\r%d-grams built: %d" % (n, i)

        self.ngrams = ngrams
        self.ngram_map = ngram_map

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.word_map

    def indices(self, tokens):
        return [self.word_map[token] if token in self else self.word_map['{rare}'] for token in tokens]


class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constants

        table_size = 1e8
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0 # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power))/norm
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
    # fo.write('%d %d\n' % (len(nn0), dim))
    for token, vector in zip(vocab, nn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

if __name__ == '__main__':

    # Number of negative examples
    k_negative_sampling = 5

    # Dimensionality of word embeddings
    dim = 100

    # Max window length
    window = 5

    # Min count for words to be used in the model, else {rare}
    min_count = 5

    # Read the corpus
    corpus = Corpus('text-aa')

    # Read train file to init vocab
    vocab = Vocabulary(corpus, min_count)

    # Initialize network
    nn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))
    nn1 = np.zeros(shape=(len(vocab), dim))

    global_word_count = 0
    table = TableForNegativeSamples(vocab)

    # Initial learning rate
    initial_alpha = 0.025

    # Modified in loop
    alpha = initial_alpha
    word_count = 0
    last_word_count = 0

    for token_idx, token in enumerate(corpus):
        if word_count % 10000 == 0:
            global_word_count += (word_count - last_word_count)
            last_word_count = word_count

            # Recalculate alpha
            alpha = initial_alpha * (1 - float(global_word_count) / len(corpus))
            if alpha < initial_alpha * 0.0001: alpha = initial_alpha * 0.0001

            sys.stdout.write("\rTraining: %d of %d" % (global_word_count, len(corpus)))
            sys.stdout.flush()

        # Randomize window size, where win is the max window size
        current_window = np.random.randint(low=1, high=window+1)
        context_start = max(token_idx - current_window, 0)
        context_end = min(token_idx + current_window + 1, len(corpus))
        context = corpus.tokens[context_start:token_idx] + corpus.tokens[token_idx+1:context_end] # Turn into an iterator?

        for context_word in context:
            # Init neu1e with zeros
            neu1e = np.zeros(dim)
            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
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
    sys.stdout.write("\rTraining: %d of %d" % (global_word_count, len(corpus)))
    sys.stdout.flush()

    # Save model to file
    save(vocab, nn0, 'output')
