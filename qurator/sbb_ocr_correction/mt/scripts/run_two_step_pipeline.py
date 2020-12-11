import io
import json
import numpy as np
import os
import pickle
import torch
import sys
home_dir = os.path.expanduser('~')

sys.path.append(home_dir + '/Qurator/mono-repo/sbb_ocr_correction/qurator/sbb_ocr_correction/data_preproc/dta')
from database import load_alignments_from_sqlite, save_alignments_to_sqlite
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

    # path definitions
    alignments_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_sliding_window_german_biased_2charges_170920.db'

    ocr_encodings_testing_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_ocr_sliding_window_3_2charges_170920.npy'
    gt_encodings_testing_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_gt_sliding_window_3_2charges_170920.npy'

    detector_token_to_code_path = home_dir + '/Qurator/used_data/features/dta/token_to_code_mapping_sliding_window_3_150620.json'#token_to_code_mapping_3_2charges_110920.json'
    detector_code_to_token_path = home_dir + '/Qurator/used_data/features/dta/code_to_token_mapping_sliding_window_3_150620.json'#code_to_token_mapping_3_2charges_110920.json'
    translator_token_to_code_path = home_dir + '/Qurator/used_data/features/dta/token_to_code_mapping_sliding_window_3_charge2_080920.json' #token_to_code_mapping_sliding_window_3_150620.json'
    translator_code_to_token_path = home_dir + '/Qurator/used_data/features/dta/code_to_token_mapping_sliding_window_3_charge2_080920.json'

    targets_testing_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_3_2charges_170920.npy'
    targets_testing_char_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_char_2charges_170920.npy'

    detector_model_path = home_dir + '/Qurator/used_data/models/models_detector_sliding_window_512_3L_LSTM_bidirec_3_070920/trained_detector_model_512_3L_LSTM_bidirec_070920_138.pt'
    translator_model_path = home_dir + '/Qurator/used_data/models/models_translator_sliding_window_256_1L_LSTM_monodirec_3_100920/trained_translator_model_256_1L_LSTM_monodirec_100920_876.pt'

    error_predictions_testing_path = home_dir + '/Qurator/used_data/output_data/predictions_pipe_detector_testing_sliding_window_512_3L_LSTM_bidirec_3_170920_138.pt'

    ocr_encodings_incorrect_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_ocr_pipe_sliding_window_3_2charges_170920.npy'
    gt_encodings_incorrect_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_gt_pipe_sliding_window_3_2charges_170920.npy'
    ocr_encodings_correct_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_ocr_pipe_sliding_window_3_2charges_170920.npy'
    gt_encodings_correct_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_gt_pipe_sliding_window_3_2charges_170920.npy'

    ocr_sequences_incorrect_path = home_dir + '/Qurator/used_data/output_data/sequences_incorrect_ocr_pipe_sliding_window_3_2charges_170920.pkl'
    gt_sequences_incorrect_path = home_dir + '/Qurator/used_data/output_data/sequences_incorrect_gt_pipe_sliding_window_3_2charges_170920.pkl'
    ocr_sequences_correct_path = home_dir + '/Qurator/used_data/output_data/sequences_correct_ocr_pipe_sliding_window_3_2charges_170920.pkl'
    gt_sequences_correct_path = home_dir + '/Qurator/used_data/output_data/sequences_correct_gt_pipe_sliding_window_3_2charges_170920.pkl'

    ocr_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_gt_pipe_sliding_window_3_2charges_hack_170920.npy'
    ocr_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_gt_pipe_sliding_window_3_2charges_hack_170920.npy'

    alignments_incorrect_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_incorrect_sliding_window_german_biased_2charges_170920.db'
    alignments_correct_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_correct_sliding_window_german_biased_2charges_170920.db'


    translator_ocr_sequences_path = home_dir + '/Qurator/used_data/features/dta/ocr_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.npy'
    translator_decoded_sequences_path = home_dir + '/Qurator/used_data/output_data/decoded_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #decoded_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    translator_pred_sequences_path = home_dir + '/Qurator/used_data/output_data/pred_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #pred_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    translator_gt_sequences_path = home_dir + '/Qurator/used_data/output_data/gt_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #gt_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'


    print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    ocr_encodings_testing = np.load(ocr_encodings_testing_path, allow_pickle=True)[:83600]
    gt_encodings_testing = np.load(gt_encodings_testing_path, allow_pickle=True)[:83600]
    assert ocr_encodings_testing.shape == gt_encodings_testing.shape

    with io.open(detector_token_to_code_path, mode='r') as f_in:
        detector_token_to_code_mapping = json.load(f_in)
    with io.open(detector_code_to_token_path, mode='r') as f_in:
        detector_code_to_token_mapping = json.load(f_in)

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(alignments_path, size='total')

    alignments = alignments[:83600]

    print('Alignments: {}'.format(len(alignments)))

    print('OCR testing encoding dimensions: {}'.format(ocr_encodings_testing.shape))
    print('GT testing encoding dimensions: {}'.format(gt_encodings_testing.shape))

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    detector_encoding_size = len(detector_token_to_code_mapping) + 1
    print('Token encodings: {}'.format(detector_encoding_size))

    detector_targets_testing = np.load(targets_testing_path)
    detector_targets_testing = detector_targets_testing[0:83600]
    print('Target testing dimensions: {}'.format(detector_targets_testing.shape))

    detector_targets_char_testing = np.load(targets_testing_char_path)
    detector_targets_char_testing = detector_targets_char_testing[0:83600]

    print('\n2. INITIALIZE DETECTOR DATASET OBJECT')

    detector_dataset_testing = OCRCorrectionDataset(ocr_encodings_testing, gt_encodings_testing)

    print('Testing size: {}'.format(len(detector_dataset_testing)))
    #print('Training size: {}'.format(len(dataset_training)))

    detector_input_size = detector_encoding_size
    detector_hidden_size = 512
    detector_output_size = 3
    detector_batch_size = 200
    detector_seq_length = detector_dataset_testing[0].shape[-1]
    detector_num_layers = 3
    detector_dropout = 0.2
    detector_bidirectional = True
    detector_activation = 'softmax'
    detector_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = detectorLSTM(detector_input_size, detector_hidden_size, detector_output_size, detector_batch_size, detector_num_layers, bidirectional=detector_bidirectional, activation=detector_activation, device=detector_device).to(detector_device)

    detector_checkpoint = torch.load(detector_model_path, map_location=detector_device)

    detector.load_state_dict(detector_checkpoint['trained_detector']) # trained_detector

    detector.eval()

    print('\n4. PREDICT ERRORS')

    error_predictions_validation = predictItersDetector(detector_dataset_testing, detector_targets_testing, detector, detector_batch_size, detector_output_size, device=detector_device)
    torch.save(error_predictions_validation, error_predictions_testing_path)
################################################################################

    # definition: conversion function
    def convert_softmax_prob_to_label(batch, threshold=0.8):
        seq_length = batch.shape[0]
        batch_size = batch.shape[1]
        batch_predicted_labels = torch.zeros([seq_length, batch_size])
        #batch_predicted_labels = torch.zeros([batch_size, seq_length])

        for si in range(seq_length):
            for bi in range(batch_size):
                max_index = torch.argmax(batch[si, bi])

                #import pdb; pdb.set_trace()
                if max_index == 2:
                    if batch[si, bi][max_index] >= threshold:
                        batch_predicted_labels[si, bi] = max_index.item()
                    else:
                        batch_predicted_labels[si, bi] = 1
                else:
                    batch_predicted_labels[si, bi] = max_index.item()
        #for bi in range(batch_size):
        #    for si in range(seq_length):
        #        max_index = torch.argmax(batch[bi, si])
        #        batch_predicted_labels[bi, si] = max_index

        return batch_predicted_labels

    target_index = 0

    predicted_labels_total = []
    predicted_labels_total = np.zeros_like(detector_targets_testing)

    batch_id = 0
    print('\n5. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')
    for predicted_batch in error_predictions_validation:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1

        #target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch, threshold=0.99)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64).numpy()

        predicted_labels_total[target_index:(target_index+detector_batch_size), :] = batch_predicted_labels

        target_index += detector_batch_size

    predicted_sequence_labels = np.zeros((len(detector_targets_testing), 1))
    for seq_i, sequence in enumerate(predicted_labels_total):
        if 2 in sequence:
            predicted_sequence_labels[seq_i] = 1
        else:
            predicted_sequence_labels[seq_i] = 0

    #import pdb; pdb.set_trace()

    target_sequence_labels = np.zeros((len(detector_targets_char_testing), 1))
    for seq_i, sequence in enumerate(detector_targets_char_testing):
        if 2 in sequence:
            target_sequence_labels[seq_i] = 1
        else:
            target_sequence_labels[seq_i] = 0

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    average = 'binary'

    f1 = f1_score(target_sequence_labels, predicted_sequence_labels, average=average)
    prec = precision_score(target_sequence_labels, predicted_sequence_labels, average=average)
    recall = recall_score(target_sequence_labels, predicted_sequence_labels, average=average)

    conf_matrix = confusion_matrix(target_sequence_labels, predicted_sequence_labels, labels=[0,1])

    print('Target sum of erroneous sequences: {}'.format(np.sum(target_sequence_labels)))
    print('Predicted sum of erroneous sequences: {}'.format(np.sum(predicted_sequence_labels)))

    print('F1 score: {}'.format(f1))
    print('Precision score: {}'.format(prec))
    print('Recall score: {}'.format(recall))

    print('Confusion matrix:\n{}'.format(conf_matrix))
################################################################################

    print('\n5. CREATE DATASET FOR TRANSLATION:')

    # based on detector output
    alignments_correct = []
    alignments_incorrect = []
    ocr_encodings_correct = []
    gt_encodings_correct = []
    ocr_encodings_incorrect = []
    gt_encodings_incorrect = []
    for alignment, sequence_label, ocr_encoding, gt_encoding in zip(alignments, target_sequence_labels, ocr_encodings_testing, gt_encodings_testing):
        if sequence_label == 1:
            ocr_encodings_incorrect.append(ocr_encoding)
            gt_encodings_incorrect.append(gt_encoding)
            alignments_incorrect.append(alignment)
        elif sequence_label == 0:
            ocr_encodings_correct.append(ocr_encoding)
            gt_encodings_correct.append(gt_encoding)
            alignments_correct.append(alignment)

    save_alignments_to_sqlite(alignments_incorrect, path=alignments_incorrect_path, append=False)
    save_alignments_to_sqlite(alignments_correct, path=alignments_correct_path, append=False)

    ocr_encodings_incorrect = np.array(ocr_encodings_incorrect)
    gt_encodings_incorrect = np.array(gt_encodings_incorrect)
    ocr_encodings_correct = np.array(ocr_encodings_correct)
    gt_encodings_correct = np.array(gt_encodings_correct)

    np.save(ocr_encodings_incorrect_path, ocr_encodings_incorrect)
    np.save(gt_encodings_incorrect_path, gt_encodings_incorrect)
    np.save(ocr_encodings_correct_path, ocr_encodings_correct)
    np.save(gt_encodings_correct_path, gt_encodings_correct)

    print('OCR encoding dimensions (incorrect): {}'.format(ocr_encodings_incorrect.shape))
    print('GT encoding dimensions (incorrect): {}'.format(gt_encodings_incorrect.shape))

    encoding_arrays = [ocr_encodings_incorrect, gt_encodings_incorrect, ocr_encodings_correct, gt_encodings_correct]

    detector_output_sequences = [[], [], [], []]

    for i, encoding_array in enumerate(encoding_arrays):
        for sequence_encoding in encoding_array:
            ocr_sequence = []
            for e in sequence_encoding:
                if e == 0:
                    break
                ocr_sequence.append(detector_code_to_token_mapping[str(e)])
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

            detector_output_sequences[i].append(''.join(ocr_sequence_filtered))

    with io.open(ocr_sequences_incorrect_path, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[0], f_out)
    with io.open(gt_sequences_incorrect_path, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[1], f_out)
    with io.open(ocr_sequences_correct_path, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[2], f_out)
    with io.open(gt_sequences_correct_path, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[3], f_out)

################################################################################

    ocr_encodings_incorrect_hack = np.load(ocr_sequences_incorrect_encoded_path)
    gt_encodings_incorrect_hack = np.load(gt_sequences_incorrect_encoded_path)
    ocr_encodings_correct_hack = np.load(ocr_sequences_correct_encoded_path)
    gt_encodings_correct_hack = np.load(gt_sequences_correct_encoded_path)

################################################################################

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

    ocr_encodings_incorrect_hack = ocr_encodings_incorrect_hack[:data_incorrect_size]
    gt_encodings_incorrect_hack = gt_encodings_incorrect_hack[:data_incorrect_size]

    translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect_hack, gt_encodings_incorrect_hack)
    
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

    ocr_cer = []
    for a in alignments:
        ocr_cer.append(a[5])

    pred_cer = []
    ocr_cer_filtered = []
    i = 0
    for pred, gt, o_cer in zip(translator_pred_sequences, translator_gt_sequences, ocr_cer):

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
    for ocr, gt, pred in zip(translator_ocr_sequences, translator_gt_sequences, translator_pred_sequences):
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
