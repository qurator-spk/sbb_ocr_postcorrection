from collections import Counter
import langid
import Levenshtein
from nltk.tokenize import WordPunctTokenizer
import re


class HeuristicsCorrector():
    def __init__(self, data, modern_dict, hist_dict, ocr_errors, lang='de'):

        self.data = data
        self.modern_dict = modern_dict
        self.hist_dict = hist_dict
        self.vocab = self.create_vocab([text for text in data if text['text'] is not None])
        self.ocr_errors = ocr_errors
        self.lang = lang
        self.token_list = self.create_token_list([text for text in data if text['text'] is not None and self.check_language(text['text'])[0] == 'de'])

    def run_correction(self):
        '''
        Run correction pipeline.

        Steps:
        0. Tokenize text.
        1. Check if token exists in dictionaries or is numeric.
        2. Check if OCR error patterns exist in token.
        3. Create alternative spelling variants based on OCR error pattern.
        4. Calculate Levenshtein distance.
        5. Decide on best candidate (based on Levenshtein distance).
        6. Correct token.
        7. Join corrected tokenized text together.
        '''
        for i, text in enumerate(self.data):
            if text['text'] is None:
                continue
            if self.lang == 'de' and self.check_language(text['text'])[0] != 'de':
                continue
            tokenized_text = self.tokenize_text(text['text'])
            for ii, token in enumerate(tokenized_text):
                if self.is_in_modern_dict(token):
                    continue
                elif self.is_in_hist_dict(token):
                    continue
                elif self.is_numeric(token):
                    continue
                else:
                    if self.is_potential_ocr_error_in_word(token):
                        alt_spellings = self.create_alternative_spellings(token)
                        distances = self.calculate_levenshtein_distance(token, alt_spellings)
                        best_candidate = self.get_best_candidate(distances)
                        tokenized_text = self.correct_token(tokenized_text, ii, best_candidate[0])
            self.data[i]['corrected_text'] = self.join_tokenized_text(tokenized_text)

    def run_frequency_calculation(self):
        '''
        Run frequency calculation.

        Steps:
        0. Loop over token list.
        1. Check if token exists in dictionaries or is numeric or alphanumeric.
        2. Create lists based on these checks.
        3. Calculate token frequencies per list.
        '''
        freq_total = self.calculate_frequencies(self.token_list)
        self.freq_total = freq_total

        tokens_in_modern_dict = []
        tokens_in_hist_dict = []
        numeric_tokens = []
        alphanumeric_tokens = []
        potential_errors = []

        for tokens in self.token_list:
            in_modern_dict = []
            in_hist_dict = []
            numeric = []
            alphanumeric = []
            errors = []
            for token in tokens:
                if self.is_in_modern_dict(token):
                    in_modern_dict.append(token)
                elif self.is_in_hist_dict(token):
                    in_hist_dict.append(token)
                elif self.is_numeric(token):
                    numeric.append(token)
                elif self.is_alphanumeric(token):
                    alphanumeric.append(token)
                else:
                    errors.append(token)
            tokens_in_modern_dict.append(in_modern_dict)
            tokens_in_hist_dict.append(in_hist_dict)
            numeric_tokens.append(numeric)
            alphanumeric_tokens.append(alphanumeric)
            potential_errors.append(errors)
        freq_modern = self.calculate_frequencies(tokens_in_modern_dict)
        self.freq_modern = freq_modern
        freq_hist = self.calculate_frequencies(tokens_in_hist_dict)
        self.freq_hist = freq_hist
        freq_numeric = self.calculate_frequencies(numeric_tokens)
        self.freq_numeric = freq_numeric
        freq_alphanumeric = self.calculate_frequencies(alphanumeric_tokens)
        self.freq_alphanumeric = freq_alphanumeric
        freq_errors = self.calculate_frequencies(potential_errors)
        self.freq_errors = freq_errors

    def calculate_frequencies(self, token_list):
        '''Calculate token frequencies.'''
        counter = Counter()
        for tokens in token_list:
            for token in tokens:
                counter[token] += 1
        return counter

    def calculate_levenshtein_distance(self, token, alt_spellings):
        '''Calculate Levenshtein distance of alternative spellings to error.'''
        distances = {}
        for spelling in alt_spellings:
            distances[spelling] = Levenshtein.distance(token, spelling)
        return distances

    def check_language(self, text):
        '''Caculate language probability.'''
        return langid.classify(text)

    def correct_token(self, tokenized_text, index, best_candidate):
        '''Replace error with substitute.'''
        tokenized_text[index] = best_candidate
        return tokenized_text

    def create_alternative_spellings(self, token):
        '''Create alternative spellings of incorrect word.'''

        def get_potential_multiletter_error(token, index):
            for i in list(range(index, len(token))):
                if not index == i:
                    if token[index:i+1] in self.ocr_errors.keys():
                        return token[index:i+1]
            return None

        def replace_str_by_index(token, index, sub, conversion):
            if conversion == '1to1':
                if index == 0:
                    return '%s%s' % (sub, token[index+1:])
                else:
                    return '%s%s%s' % (token[:index], sub, token[index+1:])
            elif conversion == '1tomany':
                if index == 0:
                    return '%s%s' % (sub, token[index+1:])
                else:
                    return '%s%s%s' % (token[:index], sub, token[index+1:])
            elif conversion == 'manyto1':
                if index == 0:
                    return '%s%s' % (sub, token[index+len(sub)+1:])
                else:
                    return '%s%s%s' % (token[:index], sub, token[index+len(sub)+1:])
            elif conversion == 'manytomany':
                if index == 0:
                    return '%s%s' % (sub, token[index+len(sub):])
                else:
                    return '%s%s%s' % (token[:index], sub, token[index+len(sub):])

        alternative_spellings = []
        for i, letter in enumerate(token):
            if letter in self.ocr_errors.keys():
                if len(self.ocr_errors[letter]) == 1:
                    conversion = '1to1'
                elif len(self.ocr_errors[letter]) > 1:
                    conversion = '1tomany'
                alternative_spellings.append(replace_str_by_index(token, i, self.ocr_errors[letter], conversion=conversion))
            multi_grapheme = get_potential_multiletter_error(token, i)
            if multi_grapheme:
                if len(self.ocr_errors[multi_grapheme]) == 1:
                    conversion = 'manyto1'
                elif len(self.ocr_errors[multi_grapheme]) > 1:
                    conversion = 'manytomany'
                alternative_spellings.append(replace_str_by_index(token, i, self.ocr_errors[multi_grapheme], conversion=conversion))

        return alternative_spellings

    def create_token_list(self, data):
        '''Create token list (using tokenizer)'''
        token_list = []
        for text in data:
            token_list.append(self.tokenize_text(text['text']))
        return token_list

    def create_vocab(self, data):
        '''Create corpus vocabulary (using tokenizer)'''
        vocab = set()
        for text in data:
            tokens = self.tokenize_text(text['text'])
            for token in tokens:
                vocab.add(token)
        return vocab

    def get_best_candidate(self, distances):
        '''Choose best candidate (the one with minimal Levenshtein distance).'''
        min_key = min(distances, key=distances.get)
        return (min_key, distances[min_key])

    def is_alphanumeric(self, token):
        '''Check if token is alphanumeric (contains BOTH numeric and alphabetic characters).'''
        return True if re.match('^(?=.*[0-9])(?=.*[a-zA-Z])', token) else False

    def is_in_hist_dict(self, token):
        '''Check if token is in historic dictionary.'''
        return True if token in self.hist_dict else False

    def is_in_modern_dict(self, token):
        '''Check if token is in modern dictionary.'''
        return True if token in self.modern_dict else False

    #def is_numeric(self, token):
    #    return True if (len(re.findall(r'\d+', token)) > 0) else False

    def is_numeric(self, token):
        '''Check if token is numeric.'''
        # FIXME: probably needs some fixing to avoid punctuation matching
        return True if re.match('^[\d\-\.\,]+$', token) else False

    def is_potential_ocr_error_in_word(self, token):
        '''Check if OCR error occurs in token.'''
        return True if any(key in token for key in self.ocr_errors.keys()) else False

    def join_tokenized_text(self, tokenized_text):
        '''
        Join tokenized text.

        A simple join() cannot be used as this would treat alphanumeric tokens
        and punctuation the same. join_punctuation concatenates defined
        punctuation directly to previous token. Afterwards, a standard join()
        is applied.
        '''
        def join_punctuation(tokenized_text, char=',;.!?'):
            '''
            '''
            tokenized_text = iter(tokenized_text)
            current_item = next(tokenized_text)

            for item in tokenized_text:
                if item in char:
                    current_item += item
                else:
                    yield current_item
                    current_item = item
            yield current_item

        return ' '.join(join_punctuation(tokenized_text))

    def tokenize_text(self, text):
        '''
        Tokenize text.

        The WordPunctTokenizer separates punctuation from alphanumeric tokens.
        '''
        # FIXME: simplest way of tokenizing, should be changed to more sophisticated solution
        # text = re.findall(r"[\w]+", text)
        text = WordPunctTokenizer().tokenize(text)
        return text
