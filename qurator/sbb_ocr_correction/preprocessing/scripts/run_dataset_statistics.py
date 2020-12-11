from collections import Counter
import io
import os
home_dir = os.path.expanduser('~')
import pickle
import pprint
import re
import statistics
import sys

sys.path.append(home_dir + '/Qurator/mono-repo/dinglehopper/qurator')
from dinglehopper.align import align

sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/data_preproc/dta')
from database import load_alignments_from_sqlite


if __name__ == '__main__':

    alignments_training_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_00-10_sliding_window_german_biased_150620.db'
    alignments_testing_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_00-10_sliding_window_german_biased_150620.db'
    pred_sequences_path = home_dir + '/Qurator/used_data/output_data/pred_sequences_250620_500.pkl'
    gt_sequences_path = home_dir + '/Qurator/used_data/output_data/gt_sequences_250620_500.pkl'

    alignments_training, alignments_training_as_df, alignments_headers = load_alignments_from_sqlite(path=alignments_training_path, size='total')
    alignments_testing, alignments_testing_as_df, alignments_headers = load_alignments_from_sqlite(path=alignments_testing_path, size='total')

    with io.open(pred_sequences_path, mode='rb') as f_in:
        pred_sequences = pickle.load(f_in)
    with io.open(gt_sequences_path, mode='rb') as f_in:
        gt_sequences = pickle.load(f_in)

    pp = pprint.PrettyPrinter()

    print('\nDATASET STATISTICS')

    print('\n1. TOKEN STATISTICS')

    # token statistics (total)

    training_tokens = 0
    testing_tokens = 0

    for alignment in alignments_training:
        training_tokens += len(alignment[4])

    print('\nNumber of training tokens: {}'.format(training_tokens))

    for alignment in alignments_testing:
        testing_tokens += len(alignment[4])

    print('Number of testing tokens: {}'.format(testing_tokens))

    total_tokens = training_tokens + testing_tokens

    print('Number of total tokens: {}'.format(total_tokens))


    # token statistics (no whitespace)

    training_tokens_no_whitespace = 0
    testing_tokens_no_whitespace = 0

    for alignment in alignments_training:
        training_tokens_no_whitespace += len(re.sub(r'\s+', '', alignment[4]))

    print('\nNumber of training tokens after whitespace removal: {}'.format(training_tokens_no_whitespace))

    for alignment in alignments_testing:
        testing_tokens_no_whitespace += len(re.sub(r'\s+', '', alignment[4]))

    print('Number of testing tokens after whitespace removal: {}'.format(testing_tokens_no_whitespace))

    total_tokens_no_whitespace = training_tokens_no_whitespace + testing_tokens_no_whitespace

    print('Number of total tokens after whitespace removal: {}'.format(total_tokens_no_whitespace))


    # tokens statistics (only whitespace)

    training_whitespace = training_tokens - training_tokens_no_whitespace
    testing_whitespace = testing_tokens - testing_tokens_no_whitespace

    total_whitespace = training_whitespace + testing_whitespace

    print('\nNumber of training whitespace: {}'.format(training_whitespace))
    print('Number of testing whitespace: {}'.format(testing_whitespace))
    print('Number of total whitespace: {}'.format(total_whitespace))


    print('\n2. ERROR STATISTICS')

    # error statistics (sequence based)
    erroneous_sequences_training = 0
    correct_sequences_training = 0
    cer = []

    for alignment in alignments_training:
        cer.append(alignment[5])
        if alignment[3] == alignment[4]:
            correct_sequences_training += 1
        else:
            erroneous_sequences_training += 1

    total_sequences_training = correct_sequences_training + erroneous_sequences_training
    mean_cer = statistics.mean(cer)

    print('\nNumber of correct training sequences: {} | {}'.format(correct_sequences_training, round(correct_sequences_training / total_sequences_training, 3)))
    print('Number of erroneous training sequences: {} | {}'.format(erroneous_sequences_training, round(erroneous_sequences_training / total_sequences_training, 3)))
    print('Mean training CER: {}'.format(round(mean_cer, 4)))

    # error statistics (before post-correction)
    erroneous_chars_testing = 0
    correct_chars_testing = 0

    error_counter_testing_pre = Counter()

    for alignment_testing in alignments_testing:
        aligned_sequence = align(alignment_testing[3], alignment_testing[4])

        for ocr_char, gt_char in aligned_sequence:
            if ocr_char != gt_char:
                error_counter_testing_pre[str(ocr_char) + ':' + str(gt_char)] += 1
                erroneous_chars_testing += 1
            else:
                correct_chars_testing += 1

    most_common_errors_pre = []

    for error in error_counter_testing_pre.most_common(100):
        if 'None' not in error[0]:
            most_common_errors_pre.append(error)

    print('\nMost common errors before post-correction:')
    pp.pprint(most_common_errors_pre)

    total_chars_testing = correct_chars_testing + erroneous_chars_testing

    erroneous_chars_training = 0
    correct_chars_training = 0

    for alignment_training in alignments_training:
        aligned_sequence = align(alignment_training[3], alignment_training[4])

        for ocr_char, gt_char in aligned_sequence:
            if ocr_char != gt_char:
                erroneous_chars_training += 1
            else:
                correct_chars_training += 1

    total_chars_training = correct_chars_training + erroneous_chars_training

    total_erroneous_chars = erroneous_chars_training + erroneous_chars_testing
    total_correct_chars = correct_chars_training + correct_chars_testing
    total_chars = total_chars_training + total_chars_testing

    print('\nTotal number of errors: {} | {}'.format(total_erroneous_chars, round(total_erroneous_chars / total_chars, 3)))
    print('Total number of correct characters: {} | {}'.format(total_correct_chars, round(total_correct_chars / total_chars, 3)))

    erroneous_chars_testing_post = 0
    correct_chars_testing_post = 0

    error_counter_testing_post = Counter()

    for pred, gt in zip(pred_sequences, gt_sequences):
        aligned_sequence = align(pred, gt)

        for ocr_char, gt_char in aligned_sequence:
            if ocr_char != gt_char:
                error_counter_testing_post[str(ocr_char) + ':' + str(gt_char)] += 1
                erroneous_chars_testing_post += 1
            else:
                correct_chars_testing_post += 1

    most_common_errors_post = []

    for error in error_counter_testing_post.most_common(100):
        if 'None' not in error[0]:
            most_common_errors_post.append(error)

    print('\nMost common errors after post-correction:')
    pp.pprint(most_common_errors_post)

    most_common_uncorrected_errors = []
    most_common_corrected_errors = []

    uncorrected = False
    for error_pre in most_common_errors_pre:
        for error_post in most_common_errors_post:
            if error_pre[0] == error_post[0]:
                most_common_uncorrected_errors.append((error_pre[0], error_pre[1], error_post[1]))
                uncorrected = True
                break
        if not uncorrected:
            most_common_corrected_errors.append(error_pre)
        uncorrected = False


    print('\nMost common (still) uncorrected errors after post-correction:')
    pp.pprint(most_common_uncorrected_errors)
