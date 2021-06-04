import numpy as np


def encode_sequence(sequence, encoding_mapping):
    '''
    Encode single sequence.

    Keyword arguments:
    sequence (list) -- the sequence to be encoded
    encoding_mapping (dict) -- the token-code mapping

    Outputs:
    encoded_sequence (list) -- the encoded sequence
    '''

    encoded_sequence = []
    for token in sequence:
        encoded_sequence.append(encoding_mapping[token])

    return encoded_sequence


def decode_sequence(sequence, decoding_mapping):
    '''
    Decode single sequence.

    Keyword arguments:
    sequence (list) -- an encoded sequence
    decoding_mapping (dict) -- the code-token mapping

    Outputs:
    decoded_sequence (list) -- the decoded sequence
    '''
    decoded_sequence = []
    for code in sequence:
        if code == 0:
            continue
        decoded_sequence.append(decoding_mapping[str(code)])
    joined_sequence = join_sequence(decoded_sequence)
    return decoded_sequence, joined_sequence


def join_sequence(sequence):
    '''
    Join single sequence and replace WSC, SOS, and EOS.

    Keyword arguments:
    sequence (list) -- the sequence

    Outputs:
    joined_sequence (str) -- the joined sequence
    '''
    joined_sequence = ''.join(sequence)
    joined_sequence = joined_sequence.replace('<WSC>', ' ')
    joined_sequence = joined_sequence.replace('<SOS>', '')
    joined_sequence = joined_sequence.replace('<EOS>', '')

    return joined_sequence


def vectorize_encoded_sequences(encodings):
    '''
    Vectorize encoded sequence, i.e. convert to NP array. 

    Keyword arguments:
    encodings (list) -- a list of encodings

    Outputs:
    vectorized_encodings (numpy.ndarray) -- the encoding array 
    '''

    vectorized_encodings = []

    for i, encoding in enumerate(encodings):
        encoding = np.array([encoding])
        print(i)
        print(encoding)
        #vectorized_encoding = encoding.reshape(-1, len(encoding))
        vectorized_encodings.append(encoding)

    return vectorized_encodings


def add_padding(encodings, max_length):
    '''
    Add padding to sequence encodings of different lengths.

    Keyword arguments:
    encodings (list) -- the sequence encodings
    max_length (int) -- the maximal length of padded sequence

    Outputs:
    padded_encodings (numpy.ndarray) -- the padded array
    '''

    if isinstance(encodings, list):
        padded_encodings = np.zeros([len(encodings), max_length], dtype=np.int64)

        for i, encoding in enumerate(encodings):
            padded_encodings[i, :len(encoding)] = encoding
    else:
        raise TypeError("Input has {}; needs to have <class 'list'> instead".format(type(encodings)))

    return padded_encodings


def create_encoding_mappings(wordpiece_vocab, token_threshold):
    '''
    Create the vocabulary encoding.

    Keyword arguments:
    wordpiece_vocab (dict)
    token_threshold (int): the threshold of most common tokens

    Outputs:
    token_to_code_mapping (dict): the token (vocabulary)-to-code mapping
    code_to_token_mapping (dict): the code-to-token mapping
    '''

    token_to_code_mapping = {}
    code_to_token_mapping = {}

    token_to_code_mapping['<SOS>'] = 1  # Start-Of-Sentence
    token_to_code_mapping['<EOS>'] = 2  # End-Of-Sentence
    token_to_code_mapping['<WSC>'] = 3  # WhiteSpace-Character
    token_to_code_mapping['<UNK>'] = 4  # UnKnown-Character
    code_to_token_mapping[1] = '<SOS>'
    code_to_token_mapping[2] = '<EOS>'
    code_to_token_mapping[3] = '<WSC>'
    code_to_token_mapping[4] = '<UNK>'
    code = 5

    for token_length, vocab_set in wordpiece_vocab.items():
        if token_length == 1:
            for token in vocab_set:
                token_to_code_mapping[token] = code
                code_to_token_mapping[code] = token
                code += 1
        else:
            if token_threshold:
                most_common_tokens = vocab_set.most_common(token_threshold)
                for token in most_common_tokens:
                    token_to_code_mapping[token[0]] = code
                    code_to_token_mapping[code] = token[0]
                    code += 1
            else:
                for token in vocab_set:
                    token_to_code_mapping[token] = code
                    code_to_token_mapping[code] = token
                    code += 1

    assert len(token_to_code_mapping) == len(code_to_token_mapping)

    return token_to_code_mapping, code_to_token_mapping


def find_longest_sequence(ocr_encodings, gt_encodings):
    '''
    Find longest sequence in OCR and GT encodings.

    Keyword arguments:
    ocr_encodings (list) -- the OCR encodings
    gt_encodings (list) -- the GT encodings

    Outputs:
    the maximum sequence length
    '''
    ocr_max_length = max([len(encoding) for encoding in ocr_encodings])
    gt_max_length = max([len(encoding) for encoding in gt_encodings])

    return max(ocr_max_length, gt_max_length)
