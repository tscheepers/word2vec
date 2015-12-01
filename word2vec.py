import math
import sys
import numpy as np


class Ngram:
    def __init__(self, tokens):
        self.tokens = tokens
        self.count = 0
        self.score = 0.0

    def set_score(self, score):
        self.score = score

    def get_string(self):
        return ' '.join(self.tokens)


class Corpus:
    def __init__(self, filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold, word_phrase_filename):
        i = 0
        file_pointer = open(filename, 'r')

        all_tokens = []

        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                token = token.lower()
                all_tokens.append(token)

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpus: %d" % i)

        sys.stdout.flush()
        print "\rCorpus read: %d" % i

        file_pointer.close()

        self.tokens = all_tokens

        for x in range(1, word_phrase_passes + 1):
            self.build_ngrams(x, word_phrase_delta, word_phrase_threshold, word_phrase_filename)

    def build_ngrams(self, x, word_phrase_delta, word_phrase_threshold, word_phrase_filename):

        ngrams = []
        ngram_map = {}

        token_count_map = {}
        for token in self.tokens:
            if token not in token_count_map:
                token_count_map[token] = 1
            else:
                token_count_map[token] += 1

        i = 0
        ngram_l = []
        for token in self.tokens:

            if len(ngram_l) == 2:
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
                sys.stdout.write("\rBuilding n-grams (%d pass): %d" % (x, i))

        sys.stdout.flush()
        print "\rn-grams (%d pass) built: %d" % (x, i)

        filtered_ngrams_map = {}
        file_pointer = open(word_phrase_filename + ('-%d' % x), 'w')

        # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        i = 0
        for ngram in ngrams:
            product = 1
            for word_string in ngram.tokens:
                product *= token_count_map[word_string]
            ngram.set_score((float(ngram.count) - word_phrase_delta) / float(product))

            if ngram.score > word_phrase_threshold:
                filtered_ngrams_map[ngram.get_string()] = ngram
                file_pointer.write('%s %d\n' % (ngram.get_string(), ngram.count))

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rScoring n-grams: %d" % i)

        sys.stdout.flush()
        print "\rScored n-grams: %d, filtered n-grams: %d" % (i, len(filtered_ngrams_map))
        file_pointer.close()

        # Combining the tokens
        all_tokens = []
        i = 0

        while i < len(self.tokens):

            if i + 1 < len(self.tokens):
                ngram_l = []
                ngram_l.append(self.tokens[i])
                ngram_l.append(self.tokens[i+1])
                ngram_string = ' '.join(ngram_l)

                if len(ngram_l) == 2 and (ngram_string in filtered_ngrams_map):
                    ngram = filtered_ngrams_map[ngram_string]
                    all_tokens.append(ngram.get_string())
                    i += 2
                else:
                    all_tokens.append(self.tokens[i])
                    i += 1
            else:
                all_tokens.append(self.tokens[i])
                i += 1

        print "Tokens combined"

        self.tokens = all_tokens

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, corpus, min_count):
        self.words = []
        self.word_map = {}
        self.build_words(corpus, min_count)

        self.filter_for_rare_and_common()

    def build_words(self, corpus, min_count):
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
                sys.stdout.write("\rBuilding vocabulary: %d" % len(words))

        sys.stdout.flush()
        print "\rVocabulary built: %d" % len(words)

        self.words = words
        self.word_map = word_map # Mapping from each token to its index in vocab

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

    def filter_for_rare_and_common(self):
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
        pass


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


def save(vocab, nn0, filename):
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        file_pointer.write('%s %s\n' % (word, vector_str))
    file_pointer.close()


if __name__ == '__main__':

    # Number of negative examples
    k_negative_sampling = 5

    # Dimensionality of word embeddings
    dim = 100

    # Max window length
    window = 5

    # Min count for words to be used in the model, else {rare}
    min_count = 2

    # Number of word phrase passes
    word_phrase_passes = 2

    # min count for word phrase formula
    word_phrase_delta = 3

    # Threshold for something
    word_phrase_threshold = 1e-3

    # Read the corpus
    corpus = Corpus('input-10000', word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases')

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

    tokens = vocab.indices(corpus)

    for token_idx, token in enumerate(tokens):
        if word_count % 10000 == 0:
            global_word_count += (word_count - last_word_count)
            last_word_count = word_count

            # Recalculate alpha
            alpha = initial_alpha * (1 - float(global_word_count) / len(corpus))
            if alpha < initial_alpha * 0.0001:
                alpha = initial_alpha * 0.0001

            sys.stdout.flush()
            sys.stdout.write("\rTraining: %d of %d" % (global_word_count, len(corpus)))

        # Randomize window size, where win is the max window size
        current_window = np.random.randint(low=1, high=window+1)
        context_start = max(token_idx - current_window, 0)
        context_end = min(token_idx + current_window + 1, len(tokens))
        context = tokens[context_start:token_idx] + tokens[token_idx+1:context_end] # Turn into an iterator?

        for context_word in context:
            # Init neu1e with zeros
            neu1e = np.zeros(dim)
            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
            for target, label in classifiers:
                z = np.dot(nn0[context_word], nn1[target])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * nn1[target]              # Error to backpropagate to nn0
                nn1[target] += g * nn0[context_word]  # Update nn1

            # Update nn0
            nn0[context_word] += neu1e

        word_count += 1

    global_word_count += (word_count - last_word_count)
    sys.stdout.flush()
    print "\rTraining finished: %d" % global_word_count

    # Save model to file
    save(vocab, nn0, 'output')
