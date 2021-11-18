import click
import io
import json
import numpy as np
import os

from .encoding import add_padding, create_encoding_mappings, encode_sequence, find_longest_sequence, vectorize_encoded_sequences
from .tokenization import WordpieceTokenizer
from .wordpiece import WordpieceVocabGenerator

from qurator.sbb_ocr_postcorrection.preprocessing.database import load_alignments_from_sqlite


@click.command()
@click.argument('training-dir', type=click.Path(exists=True))
@click.argument('testing-dir', type=click.Path(exists=True))
@click.argument('validation-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True))
@click.argument('code-to-token-dir', type=click.Path(exists=True))
@click.option('--max-len', default=3, help='The maximal length of a to be encoded character group.')
def create_encoding_mapping(training_dir, testing_dir, validation_dir,
                            token_to_code_dir, code_to_token_dir, max_len):
    '''
    Arguments:
    training-dir --
    testing-dir --
    validation-dir --
    token-to-code-dir --
    code-to-token-dir --
    '''

#    add_another_charge = True

    # make paths absolute
    training_dir = os.path.abspath(training_dir)
    testing_dir = os.path.abspath(testing_dir)
    validation_dir = os.path.abspath(validation_dir)

    training_data, _, _ = load_alignments_from_sqlite(training_dir)
    testing_data, _, _ = load_alignments_from_sqlite(testing_dir)
    validation_data, _, _ = load_alignments_from_sqlite(validation_dir)

    aligned_data = training_data + testing_data + validation_data
#    print('\nCorpus size (1 charge): {}'.format(len(aligned_data)))
    print('\nCorpus size: {}'.format(len(aligned_data)))

#    if add_another_charge:
#        training_charge2_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_german_charge2_full_110920.db'
#        testing_charge2_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_german_charge2_full_110920.db'
#        validation_charge2_path = home_dir + '/Qurator/used_data/preproc_data/dta/validation_set_german_charge2_full_110920.db'
#
#        training_charge2_data, _, _ = load_alignments_from_sqlite(training_charge2_path)
#        testing_charge2_data, _, _ = load_alignments_from_sqlite(testing_charge2_path)
#        validation_charge2_data, _, _ = load_alignments_from_sqlite(validation_charge2_path)
#
#        aligned_data.extend(training_charge2_data)
#        aligned_data.extend(testing_charge2_data)
#        aligned_data.extend(validation_charge2_data)
#
#        print('Corpus size (2 charges): {}'.format(len(aligned_data)))

    g = WordpieceVocabGenerator(max_wordpiece_length=max_len)
    g.generate_vocab_counts(aligned_data)
    wordpiece_vocab = g.wordpiece_vocab

    token_to_code_mapping, code_to_token_mapping = create_encoding_mappings(wordpiece_vocab, token_threshold=None)

    with io.open(token_to_code_dir, mode='w') as f_out:
        json.dump(token_to_code_mapping, f_out)
    with io.open(code_to_token_dir, mode='w') as f_out:
        json.dump(code_to_token_mapping, f_out)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True))
@click.option('--seq-len', default=40, help='The maximal length of a sequence.')
def encode_features_for_single_page(in_dir, out_dir, token_to_code_dir, seq_len):
    '''
    '''
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)

    encoded_ocr_dir = os.path.join(out_dir, 'encoded_ocr.npy')

    with io.open(in_dir, mode='r') as f_in:
        page_ocr = json.load(f_in)

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)

    pad_encoding = True # unpadded version does not work yet

    tok = WordpieceTokenizer(token_to_code_mapping, token_delimiter="<WSC>", unknown_char="<UNK>")

    print_examples = True

    if seq_len is None:
        ocr_encodings = []

        for i, line in enumerate(page_ocr['none']['P0001'][0]):
            tokenized_ocr = tok.tokenize(alignment[1], print_examples)

            ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)

            ocr_encodings.append(ocr_encoding)

        #seq_len = find_longest_sequence(ocr_encodings, gt_encodings)
        #print('Max Length: {}'.format(seq_len))
    else:
        print('Max Length: {}'.format(seq_len))

    ocr_encodings = []

    for i, line in enumerate(page_ocr['none']['P0001'][0]):
    #    import pdb; pdb.set_trace()
        tokenized_ocr = tok.tokenize(line[1], print_examples)

        ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)

        ocr_encodings.append(ocr_encoding)

    if pad_encoding:
        try:
            ocr_encodings = add_padding(ocr_encodings, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            ocr_encodings = vectorize_encoded_sequences(ocr_encodings)
        except:
            pass

    np.save(encoded_ocr_dir, ocr_encodings)



################################################################################
@click.command()
@click.argument('training-dir', type=click.Path(exists=True))
@click.argument('testing-dir', type=click.Path(exists=True))
@click.argument('testing-small-dir', type=click.Path(exists=True))
@click.argument('validation-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.option('--seq-len', default=40, help='The maximal length of a sequence.')
@click.option('--exp/--no-exp', default=False, help='If exp, print examples.')
def encode_features_for_splitted_data(training_dir, testing_dir, testing_small_dir, validation_dir,
                    token_to_code_dir, out_dir, seq_len):
    '''
    Arguments:
    training-dir --
    testing-dir --
    testing-small-dir --
    validation-dir --
    token-to-code-dir --
    out-dir --
    '''

    # make paths absolute
    training_dir = os.path.abspath(training_dir)
    testing_dir = os.path.abspath(testing_dir)
    testing_small_dir = os.path.abspath(testing_small_dir)
    validation_dir = os.path.abspath(validation_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)
    out_dir = os.path.abspath(out_dir)

    encoded_training_ocr_dir = os.path.join(out_dir, 'encoded_training_ocr.npy')
    encoded_training_gt_dir = os.path.join(out_dir, 'encoded_training_gt.npy')
    encoded_testing_ocr_dir = os.path.join(out_dir, 'encoded_testing_ocr.npy')
    encoded_testing_gt_dir = os.path.join(out_dir, 'encoded_testing_gt.npy')
    encoded_testing_ocr_small_dir = home_dir + os.path.join(out_dir, 'encoded_testing_ocr_small.npy')
    encoded_testing_gt_small_dir = home_dir + os.path.join(out_dir, 'encoded_testing_gt_small.npy')
    encoded_validation_ocr_dir = home_dir + os.path.join(out_dir, 'encoded_validation_ocr.npy')
    encoded_validation_gt_dir = home_dir + os.path.join(out_dir, 'encoded_validation_gt.npy')

    training_data, _, _ = load_alignments_from_sqlite(training_dir)
    testing_data, _, _ = load_alignments_from_sqlite(testing_dir)
    testing_data_small, _, _ = load_alignments_from_sqlite(testing_small_dir)
    validation_data, _, _ = load_alignments_from_sqlite(validation_dir)

    print('\nTraining size: {}'.format(len(training_data)))
    print('Testing size: {}'.format(len(testing_data)))
    print('Testing size small: {}'.format(len(testing_data_small)))
    print('Validation size: {}'.format(len(validation_data)))

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)

    if exp:
        print_examples = True
    else:
        print_examples = False

    pad_encoding = True # unpadded version does not work yet

    tok = WordpieceTokenizer(token_to_code_mapping, token_delimiter="<WSC>", unknown_char="<UNK>")


    # Encoding Pipeline for whole dataset (to get longest sequence)
    # only needed if seq_len is not defined
    if seq_len is None:
        ocr_encodings = []
        gt_encodings = []

        for i, alignment in enumerate(aligned_data):
            tokenized_ocr = tok.tokenize(alignment[3], print_examples)
            tokenized_gt = tok.tokenize(alignment[4], print_examples)

            ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
            gt_encoding = encode_sequence(tokenized_gt, token_to_code_mapping)

            ocr_encodings.append(ocr_encoding)
            gt_encodings.append(gt_encoding)

            if i % 50000 == 0:
                print('Files: {}'.format(i))

        seq_len = find_longest_sequence(ocr_encodings, gt_encodings)
        print('Max Length: {}'.format(seq_len))
    else:
        print('Max Length: {}'.format(seq_len))

    # TRAINING SET
    training_ocr_encodings = []
    training_gt_encodings = []

    for i, alignment in enumerate(training_data):
    #    import pdb; pdb.set_trace()
        tokenized_ocr = tok.tokenize(alignment[3], print_examples)
        tokenized_gt = tok.tokenize(alignment[4], print_examples)

        ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
        gt_encoding = encode_sequence(tokenized_gt, token_to_code_mapping)

        training_ocr_encodings.append(ocr_encoding)
        training_gt_encodings.append(gt_encoding)

        if (i+1) % 50000 == 0:
            print('Training Files: {}'.format(i+1))


    if pad_encoding:
        try:
            training_ocr_encodings = add_padding(training_ocr_encodings, seq_len)
            training_gt_encodings = add_padding(training_gt_encodings, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            training_ocr_encodings = vectorize_encoded_sequences(training_ocr_encodings)
            training_gt_encodings = vectorize_encoded_sequences(training_gt_encodings)
        except:
            pass

    np.save(encoded_training_ocr_dir, training_ocr_encodings)
    np.save(encoded_training_gt_dir, training_gt_encodings)


    # TESTING SET

    testing_ocr_encodings = []
    testing_gt_encodings = []

    for i, alignment in enumerate(testing_data):
        tokenized_ocr = tok.tokenize(alignment[3], print_examples)
        tokenized_gt = tok.tokenize(alignment[4], print_examples)

        ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
        gt_encoding = encode_sequence(tokenized_gt, token_to_code_mapping)

        testing_ocr_encodings.append(ocr_encoding)
        testing_gt_encodings.append(gt_encoding)

        if (i+1) % 10000 == 0:
            print('Testing Files: {}'.format(i+1))

    if pad_encoding:
        try:
            testing_ocr_encodings = add_padding(testing_ocr_encodings, seq_len)
            testing_gt_encodings = add_padding(testing_gt_encodings, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            testing_ocr_encodings = vectorize_encoded_sequences(testing_ocr_encodings)
            testing_gt_encodings = vectorize_encoded_sequences(testing_gt_encodings)
        except:
            pass

    np.save(encoded_testing_ocr_dir, testing_ocr_encodings)
    np.save(encoded_testing_gt_dir, testing_gt_encodings)


    # TESTING SET SMALL

    testing_ocr_encodings_small = []
    testing_gt_encodings_small = []

    for i, alignment in enumerate(testing_data_small):
        tokenized_ocr = tok.tokenize(alignment[3], print_examples)
        tokenized_gt = tok.tokenize(alignment[4], print_examples)


        ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
        gt_encoding = encode_sequence(tokenized_gt, token_to_code_mapping)

        testing_ocr_encodings_small.append(ocr_encoding)
        testing_gt_encodings_small.append(gt_encoding)

        if (i+1) % 10000 == 0:
            print('Testing Files Small: {}'.format(i+1))

    if pad_encoding:
        try:
            testing_ocr_encodings_small = add_padding(testing_ocr_encodings_small, seq_len)
            testing_gt_encodings_small = add_padding(testing_gt_encodings_small, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            testing_ocr_encodings_small = vectorize_encoded_sequences(testing_ocr_encodings_small)
            testing_gt_encodings_small = vectorize_encoded_sequences(testing_gt_encodings_small)
        except:
            pass

    np.save(encoded_testing_ocr_small_dir, testing_ocr_encodings_small)
    np.save(encoded_testing_gt_small_dir, testing_gt_encodings_small)

    # VALIDATION SET

    validation_ocr_encodings = []
    validation_gt_encodings = []

    for i, alignment in enumerate(validation_data):
        tokenized_ocr = tok.tokenize(alignment[3], print_examples)
        tokenized_gt = tok.tokenize(alignment[4], print_examples)

        ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
        gt_encoding = encode_sequence(tokenized_gt, token_to_code_mapping)


        validation_ocr_encodings.append(ocr_encoding)
        validation_gt_encodings.append(gt_encoding)

        if (i+1) % 10000 == 0:
            print('Validation Files: {}'.format(i+1))

    if pad_encoding:
        try:
            validation_ocr_encodings = add_padding(validation_ocr_encodings, seq_len)
            validation_gt_encodings = add_padding(validation_gt_encodings, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            validation_ocr_encodings = vectorize_encoded_sequences(validation_ocr_encodings)
            validation_gt_encodings = vectorize_encoded_sequences(validation_gt_encodings)
        except:
            pass

    np.save(encoded_validation_ocr_dir, validation_ocr_encodings)
    np.save(encoded_validation_gt_dir, validation_gt_encodings)

################################################################################
@click.command()
@click.argument('ocr-incorrect-dir', type=click.Path(exists=True))
@click.argument('gt-incorrect-dir', type=click.Path(exists=True))
@click.argument('ocr-correct-dir', type=click.Path(exists=True))
@click.argument('gt-correct-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.option('--seq-len', default=40, help='The maximal length of a sequence.')
@click.option('--exp/--no-exp', default=False, help='If exp, print examples.')
def encode_features_hack(ocr_incorrect_dir, gt_incorrect_dir, ocr_correct_dir,
                         gt_correct_dir, token_to_code_dir, out_dir, seq_len,
                         exp):
    '''
    Arguments:
    ocr-incorrect-dir --
    gt-incorrect-dir --
    ocr-correct-dir --
    gt-correct-dir --
    token-to-code-dir --
    out-dir --
    '''

    # make paths absolute
    ocr_incorrect_dir = os.path.abspath(ocr_incorrect_dir)
    gt_incorrect_dir = os.path.abspath(gt_incorrect_dir)
    ocr_correct_dir = os.path.abspath(ocr_correct_dir)
    gt_correct_dir = os.path.abspath(gt_correct_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)
    out_dir = os.path.abspath(out_dir)

    ocr_incorrect_encoded_dir = os.path.join(out_dir, 'encoded_incorrect_ocr.npy')
    gt_incorrect_encoded_dir = os.path.join(out_dir, 'encoded_incorrect_gt.npy')
    ocr_correct_encoded_dir = os.path.join(out_dir, 'encoded_correct_ocr.npy')
    gt_correct_encoded_dir = os.path.join(out_dir, 'encoded_correct_gt.npy')

    with io.open(ocr_incorrect_dir, mode='rb') as f_in:
        ocr_sequences_incorrect = pickle.load(f_in)
    with io.open(gt_incorrect_dir, mode='rb') as f_in:
        gt_sequences_incorrect = pickle.load(f_in)
    with io.open(ocr_correct_dir, mode='rb') as f_in:
        ocr_sequences_correct = pickle.load(f_in)
    with io.open(gt_correct_dir, mode='rb') as f_in:
        gt_sequences_correct = pickle.load(f_in)

    print('\nOCR sequences (incorrect): {}'.format(len(ocr_sequences_incorrect)))
    print('GT sequences (incorrect): {}'.format(len(gt_sequences_incorrect)))
    print('OCR sequencens (correct): {}'.format(len(ocr_sequences_correct)))
    print('GT sequences (correct): {}'.format(len(gt_sequences_correct)))

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)

    if exp:
        print_examples = True
    else:
        print_examples = False
    pad_encoding = True # unpadded version does not work yet

    tok = WordpieceTokenizer(token_to_code_mapping, token_delimiter="<WSC>", unknown_char="<UNK>")

    print('Max Length: {}'.format(seq_len))

    ocr_encodings_incorrect = []
    for i, sequence in enumerate(ocr_sequences_incorrect):
    #    import pdb; pdb.set_trace()
        tokenized = tok.tokenize(sequence, print_examples)

        encoding = encode_sequence(tokenized, token_to_code_mapping)

        ocr_encodings_incorrect.append(encoding)

    if pad_encoding:
        try:
            ocr_encodings_incorrect = add_padding(ocr_encodings_incorrect, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            ocr_encodings_incorrect = vectorize_encoded_sequences(ocr_encodings_incorrect)
        except:
            pass
    np.save(ocr_incorrect_encoded_dir, ocr_encodings_incorrect)

    gt_encodings_incorrect = []
    for i, sequence in enumerate(gt_sequences_incorrect):
    #    import pdb; pdb.set_trace()
        tokenized = tok.tokenize(sequence, print_examples)

        encoding = encode_sequence(tokenized, token_to_code_mapping)

        gt_encodings_incorrect.append(encoding)

    if pad_encoding:
        try:
            gt_encodings_incorrect = add_padding(gt_encodings_incorrect, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            gt_encodings_incorrect = vectorize_encoded_sequences(gt_encodings_incorrect)
        except:
            pass

    np.save(gt_incorrect_encoded_dir, gt_encodings_incorrect)

    ocr_encodings_correct = []
    for i, sequence in enumerate(ocr_sequences_correct):
    #    import pdb; pdb.set_trace()
        tokenized = tok.tokenize(sequence, print_examples)

        encoding = encode_sequence(tokenized, token_to_code_mapping)

        ocr_encodings_correct.append(encoding)

    if pad_encoding:
        try:
            ocr_encodings_correct = add_padding(ocr_encodings_correct, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            ocr_encodings_correct = vectorize_encoded_sequences(ocr_encodings_correct)
        except:
            pass

    np.save(ocr_correct_encoded_dir, ocr_encodings_correct)

    gt_encodings_correct = []
    for i, sequence in enumerate(gt_sequences_correct):
    #    import pdb; pdb.set_trace()
        tokenized = tok.tokenize(sequence, print_examples)

        encoding = encode_sequence(tokenized, token_to_code_mapping)

        gt_encodings_correct.append(encoding)

    if pad_encoding:
        try:
            gt_encodings_correct = add_padding(gt_encodings_correct, seq_len)
        except TypeError as te:
            print(te)
    else:
        try:
            gt_encodings_correct = vectorize_encoded_sequences(gt_encodings_correct)
        except:
            pass

    np.save(gt_correct_encoded_dir, gt_encodings_correct)
