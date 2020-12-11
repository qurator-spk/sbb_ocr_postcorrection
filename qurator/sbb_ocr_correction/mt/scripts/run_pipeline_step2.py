import io
import json
import numpy as np
import os
import pickle
import torch
import sys
home_dir = os.path.expanduser('~')

sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/data_preproc/dta')
from database import load_alignments_from_sqlite
sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/mt/preproc')
from data import OCRCorrectionDataset
sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/mt/models')
from error_detector import detectorLSTM
from encoder import EncoderLSTM
from decoder import AttnDecoderLSTM, DecoderLSTM
from predict import predict_detector, predictItersDetector, predict, predictIters
sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/mt/feature_extraction')
from encoding import decode_sequence
sys.path.append(home_dir + '/Qurator/mono-repo/dinglehopper/qurator/dinglehopper')
import character_error_rate

sys.path.append(home_dir + '/Qurator/mono-repo/dinglehopper/qurator')
from dinglehopper.align import seq_align

if __name__ == '__main__':

    alignments_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_sliding_window_german_biased_2charges_170920.db'
    alignments_incorrect_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_incorrect_sliding_window_german_biased_2charges_170920.db'
    alignments_correct_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_correct_sliding_window_german_biased_2charges_170920.db'

    translator_model_path = home_dir + '/Qurator/used_data/models/models_translator_sliding_window_256_1L_LSTM_monodirec_3_100920/trained_translator_model_256_1L_LSTM_monodirec_100920_876.pt'

    translator_token_to_code_path = home_dir + '/Qurator/used_data/features/dta/token_to_code_mapping_sliding_window_3_charge2_080920.json' #token_to_code_mapping_sliding_window_3_150620.json'
    translator_code_to_token_path = home_dir + '/Qurator/used_data/features/dta/code_to_token_mapping_sliding_window_3_charge2_080920.json'

    ocr_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_gt_pipe_sliding_window_3_2charges_hack_170920.npy'
    ocr_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_gt_pipe_sliding_window_3_2charges_hack_170920.npy'

    translator_ocr_sequences_path = home_dir + '/Qurator/used_data/features/dta/ocr_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.npy'
    translator_decoded_sequences_path = home_dir + '/Qurator/used_data/output_data/decoded_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #decoded_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    translator_pred_sequences_path = home_dir + '/Qurator/used_data/output_data/pred_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #pred_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    translator_gt_sequences_path = home_dir + '/Qurator/used_data/output_data/gt_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #gt_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'


    ocr_encodings_incorrect = np.load(ocr_sequences_incorrect_encoded_path)
    gt_encodings_incorrect = np.load(gt_sequences_incorrect_encoded_path)
    ocr_encodings_correct = np.load(ocr_sequences_correct_encoded_path)
    gt_encodings_correct = np.load(gt_sequences_correct_encoded_path)

    ################################################################################

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(alignments_path, size='total')
    alignments = alignments[:83600]

    alignments_incorrect, _, _ = load_alignments_from_sqlite(alignments_incorrect_path, size='total')
    data_incorrect_size = len(alignments_incorrect)
    data_incorrect_size -= (data_incorrect_size % 200)
    alignments = alignments[:(len(alignments) - (len(alignments_incorrect) % 200))]

    alignments_incorrect = alignments_incorrect[:data_incorrect_size]

    alignments_correct, _, _ = load_alignments_from_sqlite(alignments_correct_path, size='total')

    print('\n3. TRANSLATOR MODEL')

    with io.open(translator_token_to_code_path, mode='r') as f_in:
        translator_token_to_code_mapping = json.load(f_in)
    with io.open(translator_code_to_token_path, mode='r') as f_in:
        translator_code_to_token_mapping = json.load(f_in)

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    translator_encoding_size = len(translator_token_to_code_mapping) + 1
    print('Token encodings: {}'.format(translator_encoding_size))

    print('\n3.1. INITIALIZE DATASET OBJECT')

    data_incorrect_size = ocr_encodings_incorrect.shape[0]
    data_incorrect_size -= (data_incorrect_size % 200)

    ocr_encodings_incorrect = ocr_encodings_incorrect[:data_incorrect_size]
    gt_encodings_incorrect = gt_encodings_incorrect[:data_incorrect_size]

    translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect, gt_encodings_incorrect)

    print('Testing size: {}'.format(len(translator_dataset_testing)))

    print('\n3.2. DEFINE HYPERPARAMETERS AND LOAD ENCODER/DECODER NETWORKS')

    translator_input_size = translator_encoding_size
    translator_hidden_size = 256
    translator_output_size = translator_input_size
    translator_batch_size = 200
    translator_seq_length = translator_dataset_testing[0].shape[-1]
    translator_num_layers = 1
    translator_dropout = 0.2
    translator_with_attention = True
    translator_teacher_forcing_ratio = 0.5
    translator_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderLSTM(translator_input_size, translator_hidden_size, translator_batch_size, translator_num_layers, device=translator_device)

    if translator_with_attention:
        decoder = AttnDecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, translator_seq_length, num_layers=translator_num_layers, dropout=translator_dropout, device=translator_device)
    else:
        decoder = DecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, device=translator_device)

    translator_checkpoint = torch.load(translator_model_path, map_location=translator_device)
    encoder.load_state_dict(translator_checkpoint['trained_encoder'])
    decoder.load_state_dict(translator_checkpoint['trained_decoder'])

    encoder.eval()
    decoder.eval()

    print('\n3.3. PREDICT SEQUENCES')

    translator_decodings = predictIters(translator_dataset_testing, encoder, decoder, translator_batch_size, translator_seq_length, translator_with_attention, device=translator_device)

    print('\n3.4. DECODE SEQUENCES')

    translator_decoded_sequences = []
    translator_pred_sequences = []
    for decoded_batch in translator_decodings:
        for decoding in decoded_batch:
            decoded_sequence, joined_sequence = decode_sequence(list(decoding), translator_code_to_token_mapping)
            translator_decoded_sequences.append(decoded_sequence)
            translator_pred_sequences.append(joined_sequence)

    with io.open(translator_decoded_sequences_path, mode='wb') as f_out:
        pickle.dump(translator_decoded_sequences, f_out)
    with io.open(translator_pred_sequences_path, mode='wb') as f_out:
        pickle.dump(translator_pred_sequences, f_out)

    translator_gt_sequences = []
    for e in gt_encodings_incorrect: # default: gt_encodings_subset
        decoded_sequence, joined_sequence = decode_sequence(list(e), translator_code_to_token_mapping)
        translator_gt_sequences.append(joined_sequence)

    with io.open(translator_gt_sequences_path, mode='wb') as f_out:
        pickle.dump(translator_gt_sequences, f_out)

    translator_ocr_sequences = []
    for o in ocr_encodings_incorrect:
        ocr_sequence = []
        for e in o:
            if e == 0:
                break
            ocr_sequence.append(translator_code_to_token_mapping[str(e)])
        ocr_sequence_filtered = []
        for o in ocr_sequence:
            if o == '<SOS>' or o == '<EOS>':
                continue
            elif o == '<WSC>':
                ocr_sequence_filtered.append(' ')
            elif o == '<UNK>':
                continue
            #elif o == 0:
            #    break
            else:
                ocr_sequence_filtered.append(o)

        translator_ocr_sequences.append(''.join(ocr_sequence_filtered))

    with io.open(translator_ocr_sequences_path, mode='wb') as f_out:
        pickle.dump(translator_ocr_sequences, f_out)
################################################################################
    print('\n3.5. COMPARISON WITH GT')

    alignments_full = alignments_correct.copy()
    alignments_full.extend(alignments_incorrect)

    full_ocr = []
    ocr_cer = []
    for a in alignments_full:
        ocr_cer.append(a[5])
        full_ocr.append([3])

    ocr_correct_sequences = []
    gt_correct_sequences = []
    for a in alignments_correct:
        ocr_correct_sequences.append(a[3])
        gt_correct_sequences.append(a[4])

    gt_incorrect_sequences = []
    for a in alignments_incorrect:
        gt_incorrect_sequences.append(a[4])

    ocr_correct_sequences.extend(translator_pred_sequences)
    gt_correct_sequences.extend(gt_incorrect_sequences)#(translator_gt_sequences)

    pred_cer = []
    ocr_cer_filtered = []
    i = 0
    for pred, gt, o_cer in zip(ocr_correct_sequences, gt_correct_sequences, ocr_cer):

        #if not (o_cer >= 0.0 and o_cer < 0.02):
        #    continue

        aligned_sequence = seq_align(pred, gt)
        aligned_pred = []
        aligned_gt = []
        for alignment in aligned_sequence:
            if alignment[0] == None:
                aligned_pred.append(' ')
            else:
                aligned_pred.append(alignment[0])
            if alignment[1] == None:
                aligned_gt.append(' ')
            else:
                aligned_gt.append(alignment[1])
        aligned_pred = ''.join(aligned_pred)
        aligned_gt = ''.join(aligned_gt)
        assert len(aligned_pred) == len(aligned_gt)

        p_cer = character_error_rate.character_error_rate(aligned_pred, aligned_gt)

        pred_cer.append(p_cer)
        ocr_cer_filtered.append(o_cer)

        if (i+1) % 10000 == 0:
            print(i+1)
        i += 1

    print('OCR CER: {}'.format(np.mean(ocr_cer_filtered)))
    print('Pred CER: {}'.format(np.mean(pred_cer)))

    print('\n3.6. FALSE CORRECTIONS RATIO')
    false_corrections_ratio = []
    for ocr, gt, pred in zip(full_ocr, gt_correct_sequences, ocr_correct_sequences):
        if not len(ocr) == len(gt) == len(pred):
            max_length = max(len(ocr), len(gt), len(pred))

            if not (len(ocr) - max_length) == 0:
                ocr += (abs((len(ocr) - max_length)) * ' ')
            if not (len(gt) - max_length) == 0:
                gt += (abs((len(gt) - max_length)) * ' ')
            if not (len(pred) - max_length) == 0:
                pred += (abs((len(pred) - max_length)) * ' ')

            assert len(ocr) == len(gt) == len(pred)

        false_corrections_count = 0
        for o, g, p in zip(ocr, gt, pred):
            if (o == g) and p != g:
                false_corrections_count += 1
        false_corrections_ratio.append(false_corrections_count/len(pred))

    print('False corrections ratio: {}'.format(np.mean(false_corrections_ratio)))
