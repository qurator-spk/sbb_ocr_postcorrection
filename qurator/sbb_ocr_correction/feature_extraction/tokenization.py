import random
import re


def whitespace_tokenize(text, insert_delimiter=None):
    '''
    A simple whitespace tokenizer.

    Keyword arguments:
    text (str) -- the text to tokenized
    insert_delimiter (str or None) -- if not None, a string to replace the
                                      whitespace with (default: None)

    Outputs:
    the tokenized text (str)
    '''

    if insert_delimiter:
        return re.split('('+insert_delimiter+')', text.replace(' ', insert_delimiter))
    else:
        return text.strip().split()


class WordpieceTokenizer(object):
    '''A tokenizer to split a sequence into wordpieces.'''

    def __init__(self, vocab, token_delimiter, unknown_char='<UNK>'):
        if isinstance(vocab, dict):
            self.vocab = vocab.keys()
        else:
            self.vocab = vocab
        self.token_delimiter = token_delimiter
        self.unknown_char = unknown_char

    def tokenize(self, sequence, print_examples=False):
        '''
        Tokenize the sequence into wordpieces.

        First, a whitespace tokenizer is applied. Start-of-Sentence (<SOS>),
        End-of-Sentence (<EOS>) and a optional token delimiter are added.
        The algorithm creates the wordpieces according to a predefined
        vocabulary. It proceeds in a greedy fashion and decreases the wordpiece
        window until only single characters are left.

        Keyword arguments:
        sequence (str) -- the sequence to be tokenized

        Outputs:
        output_tokens (list): the tokenized wordpieces + special tokens
        '''

        output_tokens = []
        output_tokens.append('<SOS>')
        for token in whitespace_tokenize(sequence, insert_delimiter=self.token_delimiter):

            if token == self.token_delimiter:
                output_tokens.append(self.token_delimiter)
                continue

            chars = list(token)
            if print_examples:
                # Random printing decision: either 1 or 0
                print_example = int(round(random.uniform(0, 1), 0))
                if print_example:
                    print('------')
                    print('Chars: ', chars)

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                #end = len(chars)
                window = 3
                #cur_substr = None
                while start < (start + window):
                    substr = token[start:(start + window)]
                    if len(substr) < window:
                        window -= 1
                        continue
                    #print(substr)
                    if substr in self.vocab:
                        #if start > 1:
                        #    sub_tokens.append('<WSC>')
                        sub_tokens.append(substr)
                        break
                    elif substr not in self.vocab and window == 1:
                        sub_tokens.append(self.unknown_char)
                        break
                    window -= 1
                if len(sub_tokens) == 0:
                    is_bad = True
                    break
                #sub_tokens.append(cur_substr)
                start += window

            if is_bad:
                #output_tokens = []
                output_tokens.append(self.unknown_char)
            else:
                if print_examples:
                    if print_example:
                        print('Tokens: ', sub_tokens)
                output_tokens.extend(sub_tokens)
        output_tokens.append('<EOS>')
        return output_tokens
