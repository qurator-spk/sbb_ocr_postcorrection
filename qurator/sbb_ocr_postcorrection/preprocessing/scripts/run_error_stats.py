from collections import Counter
import io
import os
import pprint
import pickle
import sys
home_dir = os.path.expanduser('~')

sys.path.append(home_dir + '/Qurator/mono-repo/dinglehopper/qurator')
from dinglehopper.align import align

sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/data_preproc/dta')
from database import load_alignments_from_sqlite


if __name__ == '__main__':

    alignments_path = home_dir + '/Qurator/used_data/preproc_data/dta/aligned_corpus_sliding_window_german_150620.db'
    alignments_training_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_00-10_sliding_window_german_150620.db'
    alignments_testing_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_00-10_sliding_window_german_150620.db'
    pred_sequences_path = home_dir + '/Qurator/used_data/output_data/pred_sequences_250620_500.pkl'
    #gt_sequences_path = home_dir + '/Qurator/used_data/output_data/gt_sequences_250620_500.pkl'

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(alignments_path, size='total')
    alignments_testing, alignments_testing_as_df, alignments_headers = load_alignments_from_sqlite(alignments_testing_path, size='total')

    with io.open(pred_sequences_path, mode='rb') as f_in:
        pred_sequences = pickle.load(f_in)
    #with io.open(gt_sequences_path, mode='rb') as f_in:
    #    gt_sequences = pickle.load(f_in)

    pp = pprint.PrettyPrinter()

    # Error statistics before post-correction

    error_counter_testing_pre = Counter()

    for alignment_testing in alignments_testing:
        aligned_sequence = align(alignment_testing[0], alignment_testing[1])

        for ocr_char, gt_char in aligned_sequence:
            if ocr_char != gt_char:
                error_counter_testing_pre[str(ocr_char) + ':' + str(gt_char)] += 1

    most_common_errors_pre = []

    for error in error_counter_testing_pre.most_common(100):
        if 'None' not in error[0]:
            most_common_errors_pre.append(error)

    print('\nMost common errors before post-correction:')
    pp.pprint(most_common_errors_pre)

    # Error statistics after post-correction

    error_counter_testing_post = Counter()

    for alignment_testing, pred_sequence in zip(alignments_testing[:110000], pred_sequences):
        aligned_sequence = align(alignment_testing[4], pred_sequence)

        for ocr_char, gt_char in aligned_sequence:
            if ocr_char != gt_char:
                error_counter_testing_post[str(ocr_char) + ':' + str(gt_char)] += 1

    most_common_errors_post = []

    for error in error_counter_testing_post.most_common(100):
        if 'None' not in error[0]:
            most_common_errors_post.append(error)

    print('\nMost common errors after post-correction:')
    pp.pprint(most_common_errors_post)
