from collections import Counter


class WordpieceVocabGenerator(object):
    '''The vocabulary generator needed for the wordpiece tokenizer.'''

    def __init__(self, max_wordpiece_length=3):
        self.max_wordpiece_length = max_wordpiece_length

    def whitespace_tokenize(self, text):
        '''A simple whitespace tokenizer.'''

        return text.strip().split()

    def _is_n_length(self, string, n):
        '''Checks if string is of n length.'''

        if len(string) == n:
            return True
        else:
            return False

    def generate_vocab_counts(self, corpus):
        '''
        Count the vocabulary counts in the corpus (GT and OCR).

        For GT, the wordpiece vocabulary is counted starting with
        self.max_wordpiece_length and decreasing with step size 1.

        For OCR, only single character counts are created to account for
        characters not occuring in GT.

        Keyword arguments:
        corpus (list): the corpus of alignments
        '''
        self.wordpiece_vocab = {}
        for length in range(1, self.max_wordpiece_length+1):
            if length >= 1:
                c = Counter()
                for alignment in corpus:
                    # vocab counting in GT
                    for token in self.whitespace_tokenize(alignment[4]): # alignment[4] == gt
                        for i in range(len(token)):
                            if self._is_n_length(token[i:i+length], n=length):
                                c[token[i:i+length]] += 1
                    # vocab counting in OCR
                    if length == 1:
                        for token in self.whitespace_tokenize(alignment[3]):
                            for char in token:
                                c[char] += 1
                self.wordpiece_vocab[length] = c
