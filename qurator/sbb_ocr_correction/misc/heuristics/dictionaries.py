import codecs
import pickle
import xml.etree.ElementTree as ET


def combine_dicts(dicts):
    '''Combine a list of dictionaries (sets).'''
    dict_total = set()
    for d in dicts:
        dict_total = dict_total | d

    return dict_total


def load_aspell_dict(f_path, lower=False):
    '''
    Load Aspell dictionary (modern dictionary).

    The 'lower' parameter sets the whole dictionary to lowercase.
    '''
    with codecs.open(f_path, mode='r', encoding='ISO-8859-1') as f_in:
        aspell = set()
        if lower:
            for line in f_in:
                aspell.add(line.split('/')[0].lower().strip())
        else:
            for line in f_in:
                aspell.add(line.split('/')[0].strip())
    return aspell


def load_word_inflections(f_path):
    '''Load word inflections from a German POS dict.'''
    with codecs.open(f_path, mode='r') as f_in:
        inflected_words = []
        for line in f_in:
            inflected_words += line.split()[0:2]
    return set(inflected_words)


def load_morph_dict(f_path):
    '''
    Load morphological dictionary of German.

    Source: https://github.com/DuyguA/german-morph-dictionaries
    '''
    with codecs.open(f_path, mode='r') as f_in:
        inflected_words = []
        for line in f_in:
            inflected_words.append(line.split()[0])
    return set(inflected_words)


def load_hist_dict(f_path):
    '''
    Load historical dictionary of German.

    This function loads the historical dictionary of German provided by the CIS
    OCR team (LMU Munich). They used their own XML style, so this function may
    not work for other resources.
    '''
    tree = ET.parse(f_path)
    root = tree.getroot()
    hist = set()
    for child in root:
        hist.add(child[0].text)
    return hist


def load_character_patterns(f_path):
    '''Store character patterns in a dict.'''
    with codecs.open(f_path, mode='r') as f_in:
        patterns = {}
        for line in f_in:
            variants = line.strip().split(':')
            patterns[variants[0]] = variants[1]
    return patterns


def load_pickle(f_path):
    '''Load a pickle file.'''
    with codecs.open(f_path, mode='rb') as f_in:
        return pickle.load(f_in)


class HistoricalDictionaryConverter():
    def __init__(self, modern_dict, patterns):
        '''
        Convert a modern dictionary to a historical dictionary.

        The conversion is achieved using a dict of patterns which contains
        typical historical character variants/substitutions.

        NOTE: This algorithm is naive. It creates a huge amount of false
        positives.
        '''

        self.modern_dict = modern_dict
        self.patterns = patterns

    def convert_dict(self):
        '''Convert modern dictionary to historical dictionary using patterns.'''
        hist_dict = []
        for word in self.modern_dict:
            alt_spellings = self.create_alternative_spellings(word)
            hist_dict += alt_spellings
        self.hist_dict = hist_dict

    def save_hist_dict(self, out_path):
        '''Save converted historical dictionary.'''
        with codecs.open(out_path, mode='wb') as f_out:
            pickle.dump(self.hist_dict, f_out)

    def create_alternative_spellings(self, word):
        '''Create alternative spellings of incorrect word.'''

        def get_potential_multiletter_error(word, index):
            '''
            '''
            for i in list(range(index, len(word))):
                if not index == i:
                    if word[index:i+1] in self.patterns.keys():
                        return word[index:i+1]
            return None

        def replace_str_by_index(word, index, sub, conversion):
            '''
            '''
            if conversion == '1to1':
                if index == 0:
                    return '%s%s' % (sub, word[index+1:])
                else:
                    return '%s%s%s' % (word[:index], sub, word[index+1:])
            elif conversion == '1tomany':
                if index == 0:
                    return '%s%s' % (sub, word[index+1:])
                else:
                    return '%s%s%s' % (word[:index], sub, word[index+1:])
            elif conversion == 'manyto1':
                if index == 0:
                    return '%s%s' % (sub, word[index+len(sub)+1:])
                else:
                    return '%s%s%s' % (word[:index], sub, word[index+len(sub)+1:])
            elif conversion == 'manytomany':
                if index == 0:
                    return '%s%s' % (sub, word[index+len(sub):])
                else:
                    return '%s%s%s' % (word[:index], sub, word[index+len(sub):])

        alternative_spellings = []
        for i, letter in enumerate(word):
            if letter in self.patterns.keys():
                if len(self.patterns[letter]) == 1:
                    conversion = '1to1'
                elif len(self.patterns[letter]) > 1:
                    conversion = '1tomany'
                alternative_spellings.append(replace_str_by_index(word, i, self.patterns[letter], conversion=conversion))
            multi_grapheme = get_potential_multiletter_error(word, i)
            if multi_grapheme:
                if len(self.patterns[multi_grapheme]) == 1:
                    conversion = 'manyto1'
                elif len(self.patterns[multi_grapheme]) > 1:
                    conversion = 'manytomany'
                alternative_spellings.append(replace_str_by_index(word, i, self.patterns[multi_grapheme], conversion=conversion))

        return alternative_spellings
