from collections import OrderedDict
import click
from datetime import date
import io
import json
import numpy as np
import os
import pickle
import torch

from .models.error_detector import DetectorLSTM, DetectorGRU
from .models.gan import Discriminator, Generator
from .models.predict import predict, predict_detector, predict_iters, \
    predict_iters_detector
from .models.seq2seq import AttnDecoderLSTM, DecoderLSTM, EncoderLSTM
from .models.train import  train_iters_detector, train_iters_gan, \
    train_iters_seq2seq

from qurator.sbb_ocr_postcorrection.data_structures import OCRCorrectionDataset
from qurator.sbb_ocr_postcorrection.feature_extraction.encoding import decode_sequence
from qurator.sbb_ocr_postcorrection.helpers import find_max_mod
from qurator.sbb_ocr_postcorrection.preprocessing.database import \
    load_alignments_from_sqlite
import qurator.dinglehopper.character_error_rate as character_error_rate
from qurator.dinglehopper.align import seq_align


@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('targets-char-dir', type=click.Path(exists=True))
@click.argument('error-pred-dir', type=click.Path(exists=True))
def detect_errors(ocr_dir, targets_dir, targets_char_dir, error_pred_dir):
    '''
    \b
    Arguments:
    ocr-dir --
    targets-dir --
    targets-char-dir --
    error-pred-dir --
    '''
    print('\n1. LOAD DATA (TARGETS, ERROR PREDICTIONS)')

    targets_testing = np.load(targets_testing_path)
    targets_testing = targets_testing[0:50000]
    print('Testing target dimensions: {}'.format(targets_testing.shape))

    targets_training = np.load(targets_training_path)
    targets_training = targets_training[0:365000]
    print('Trainig target dimensions: {}'.format(targets_training.shape))

    targets_char_testing = np.load(targets_testing_char_path)
    targets_char_testing = targets_char_testing[0:50000]

    error_predictions_testing = torch.load(error_predictions_testing_path)
    print('Error prediction testing dimensions: {}'.format(error_predictions_testing.shape))
    #error_predictions_training = torch.load(error_predictions_training_path)
    #print('Error prediction training dimensions: {}'.format(error_predictions_training.shape))

    ocr_encodings_testing = np.load(ocr_encodings_testing_path, allow_pickle=True)[:50000]
    print('OCR testing encodings dimensions: {}'.format(ocr_encodings_testing.shape))
    ocr_encodings_testing_flattened = ocr_encodings_testing.reshape([-1])

    ocr_encodings_training = np.load(ocr_encodings_training_path, allow_pickle=True)[:365000]
    print('OCR training encodings dimensions: {}'.format(ocr_encodings_training.shape))
    ocr_encodings_training_flattened = ocr_encodings_testing.reshape([-1])


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
    batch_size = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_labels_total = []
    predicted_labels_total = np.zeros_like(targets_testing)

    batch_id = 0
    print('\n2. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')
    for predicted_batch in error_predictions_testing:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1

        #target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch, threshold=0.99)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64).numpy()

        predicted_labels_total[target_index:(target_index+batch_size), :] = batch_predicted_labels

        target_index += batch_size

    predicted_sequence_labels = np.zeros((len(targets_testing), 1))
    for seq_i, sequence in enumerate(predicted_labels_total):
        if 2 in sequence:
            predicted_sequence_labels[seq_i] = 1
        else:
            predicted_sequence_labels[seq_i] = 0

    #import pdb; pdb.set_trace()

    target_sequence_labels = np.zeros((len(targets_char_testing), 1))
    for seq_i, sequence in enumerate(targets_char_testing):
        if 2 in sequence:
            target_sequence_labels[seq_i] = 1
        else:
            target_sequence_labels[seq_i] = 0

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    average = 'binary'

    f1 = f1_score(target_sequence_labels, predicted_sequence_labels, average=average)
    prec = precision_score(target_sequence_labels, predicted_sequence_labels, average=average)
    recall = recall_score(target_sequence_labels, predicted_sequence_labels, average=average)

    conf_matrix = confusion_matrix(target_sequence_labels, predicted_sequence_labels)

    print('Target sum of erroneous sequences: {}'.format(np.sum(target_sequence_labels)))
    print('Predicted sum of erroneous sequences: {}'.format(np.sum(predicted_sequence_labels)))

    print('F1 score: {}'.format(f1))
    print('Precision score: {}'.format(prec))
    print('Recall score: {}'.format(recall))

    print('Confusion matrix:\n{}'.format(conf_matrix))

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('pred-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('hyper-params-dir', type=click.Path(exists=True))
###OUT###
@click.argument('pred-arr-dir', type=click.Path(exists=True))
@click.argument('pred-sent-dir', type=click.Path(exists=True))
@click.argument('pred-nopad-dir', type=click.Path(exists=True))
@click.argument('targets-arr-dir', type=click.Path(exists=True))
@click.argument('targets-sent-dir', type=click.Path(exists=True))
@click.argument('targets-nopad-dir', type=click.Path(exists=True))
def evaluate_detector(ocr_dir, targets_dir, pred_dir, gt_dir, hyper_params_dir,
                      pred_arr_dir, pred_sent_dir, pred_nopad_dir,
                      targets_arr_dir, targets_sent_dir, targets_nopad_dir):
    '''
    \b
    Arguments:
    ocr-dir --
    targets-dir --
    pred-dir --
    gt-dir --
    hyper-params-dir --
    pred-arr-dir --
    pred-sent-dir --
    pred-nopad-dir --
    targets-arr-dir --
    targets-sent-dir --
    targets-nopad-dir --
    '''

    print('\n1. LOAD DATA (TARGETS, ERROR PREDICTIONS)')

    targets_testing = np.load(targets_testing_path)

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    targets_testing = targets_testing[0:size_dataset]
    print('Testing target dimensions: {}'.format(targets_testing.shape))

    #targets_training = np.load(targets_training_path)
    #targets_training = targets_training[0:365000]
    #print('Trainig target dimensions: {}'.format(targets_training.shape))

    error_predictions_testing = torch.load(error_predictions_testing_path)
    print('Error prediction testing dimensions: {}'.format(error_predictions_testing.shape))
    #error_predictions_training = torch.load(error_predictions_training_path)
    #print('Error prediction training dimensions: {}'.format(error_predictions_training.shape))

    ocr_encodings_testing = np.load(ocr_encodings_testing_path, allow_pickle=True)[:size_dataset]
    print('OCR testing encodings dimensions: {}'.format(ocr_encodings_testing.shape))
    ocr_encodings_testing_flattened = ocr_encodings_testing.reshape([-1])

    #ocr_encodings_training = np.load(ocr_encodings_training_path, allow_pickle=True)[:365000]
    #print('OCR training encodings dimensions: {}'.format(ocr_encodings_training.shape))
    #ocr_encodings_training_flattened = ocr_encodings_testing.reshape([-1])

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)

    target_index = 0
    batch_size = hyper_params['batch_size']
    device = torch.device(hyper_params['device'])

    def convert_softmax_prob_to_label(batch):
        seq_length = batch.shape[0]
        batch_size = batch.shape[1]
        batch_predicted_labels = torch.zeros([seq_length, batch_size])
        #batch_predicted_labels = torch.zeros([batch_size, seq_length])

        for si in range(seq_length):
            for bi in range(batch_size):
                max_index = torch.argmax(batch[si, bi])
                batch_predicted_labels[si, bi] = max_index
        #for bi in range(batch_size):
        #    for si in range(seq_length):
        #        max_index = torch.argmax(batch[bi, si])
        #        batch_predicted_labels[bi, si] = max_index

        return batch_predicted_labels

    ###########################
    #                         #
    # EVALUATION TESTING DATA #
    #                         #
    ###########################

    predicted_labels_total = []
    errors_per_sequence = []

    predictions = []
    targets = []

    predictions_sentence = []
    targets_sentence = []

    batch_id = 0

    print('\n2. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')
    for predicted_batch in error_predictions_testing:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1
        target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)
        target_index += batch_size

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64)

        predicted_labels_total.append(batch_predicted_labels)

        for bi in range(target_tensor.shape[0]):
            predictions.extend(batch_predicted_labels[bi])
            targets.extend(target_tensor[bi])

            if 2 in batch_predicted_labels[bi]:
                predictions_sentence.append(1)
            else:
                predictions_sentence.append(0)

            if 2 in target_tensor[bi]:
                targets_sentence.append(1)
            else:
                targets_sentence.append(0)
            #errors = 0
            #for prediction, target in zip(target_tensor[bi], batch_predicted_labels[bi]):
            #    if prediction.item() != target.item():
            #        errors += 1
            #errors_per_sequence.append(errors)

    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions_sentence = np.array(predictions_sentence)
    targets_sentence = np.array(targets_sentence)

    np.save(predictions_array_testing_path, predictions)
    np.save(targets_array_testing_path, targets)
    np.save(predictions_per_sentence_testing_path, predictions_sentence)
    np.save(targets_per_sentence_testing_path, targets_sentence)

    print('\n3. REFORMATTING PREDICTIONS WITHOUT PADDINGS')
    ocr_0_indeces = np.where(ocr_encodings_testing_flattened == 0)[0]
    predictions_no_pads = np.delete(predictions, ocr_0_indeces, axis=0)
    targets_no_pads = np.delete(targets, ocr_0_indeces, axis=0)

    np.save(predictions_no_pads_testing_path, predictions_no_pads)
    np.save(targets_no_pads_testing_path, targets_no_pads)

    #for ocr, prediction, target in zip(ocr_encodings_flattened, predictions, targets):
    #    if ocr != 0:
    #        predictions_no_pads.append(prediction.item())
    #        targets_no_pads.append(target.item())

    #with io.open(predictions_no_pads_path, mode='wb') as f_out:
    #    pickle.dump(predictions_no_pads, f_out)
    #if os.path.isfile(targets_no_pads_path):
    #    with io.open(targets_no_pads_path, mode='wb') as f_out:
    #        pickle.dump(targets_no_pads, f_out)

    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_micro = round(f1_score(targets, predictions, average='micro'),3)
    precision_micro = round(precision_score(targets, predictions, average='micro'),3)
    recall_micro = round(recall_score(targets, predictions, average='micro'),3)

    f1_weighted = round(f1_score(targets, predictions, average='weighted'),3)
    precision_weighted = round(precision_score(targets, predictions, average='weighted'),3)
    recall_weighted = round(recall_score(targets, predictions, average='weighted'),3)

    print('\nMetrics per character:')
    print('F1 (micro): {}\nPrecision (micro): {}\nRecall (micro): {}'.format(f1_micro, precision_micro, recall_micro))
    print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted, precision_weighted, recall_weighted))

    f1_weighted_sentence = round(f1_score(targets_sentence, predictions_sentence, average='weighted'),3)
    precision_weighted_sentence = round(precision_score(targets_sentence, predictions_sentence, average='weighted'),3)
    recall_weighted_sentence = round(recall_score(targets_sentence, predictions_sentence, average='weighted'),3)

    print('\nMetrics per sentence:')
    print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted_sentence, precision_weighted_sentence, recall_weighted_sentence))

    f1_weighted_no_pads = round(f1_score(targets_no_pads, predictions_no_pads, average='weighted'),3)
    precision_weighted_no_pads = round(precision_score(targets_no_pads, predictions_no_pads, average='weighted'),3)
    recall_weighted_no_pads= round(recall_score(targets_no_pads, predictions_no_pads, average='weighted'),3)

    print('\nMetrics per character (no pads):')
    print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted_no_pads, precision_weighted_no_pads, recall_weighted_no_pads))

    ############################
    #                          #
    # EVALUATION TRAINING DATA #
    #                          #
    ############################

    #predicted_labels_total_training = []
    #errors_per_sequence_training = []

    #predictions_flattened_training = []
    #targets_flattened_training = []

    #predictions_sentence_training = []
    #targets_sentence_training = []

    #batch_id = 0

    #print('\n2. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')
    #for predicted_batch in error_predictions_training:
        #print('Batch ID: {}'.format(batch_id))
    #    batch_id += 1
    #    target_tensor = torch.from_numpy(targets_training[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)
    #    target_index += batch_size

    #    batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch)
    #    batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64)

    #    predicted_labels_total_training.append(batch_predicted_labels)

    #    for bi in range(target_tensor.shape[0]):
    #        predictions_flattened_training.extend(batch_predicted_labels[bi])
    #        targets_flattened_training.extend(target_tensor[bi])

    #        if 2 in batch_predicted_labels[bi]:
    #            predictions_sentence_training.append(1)
    #        else:
    #            predictions_sentence_training.append(0)

    #        if 2 in target_tensor[bi]:
    #            targets_sentence_training.append(1)
    #        else:
    #            targets_sentence_training.append(0)
    #        #errors = 0
    #        #for prediction, target in zip(target_tensor[bi], batch_predicted_labels[bi]):
    #        #    if prediction.item() != target.item():
    #        #        errors += 1
    #        #errors_per_sequence.append(errors)
#
    #predictions_training = np.array(predictions_flattened_training)
    #targets_training = np.array(targets_flattened_training)
    #predictions_sentence_training = np.array(predictions_sentence_training)
    #targets_sentence_training = np.array(targets_sentence_training)

    #np.save(predictions_array_training_path, predictions_flattened_training)
    #np.save(targets_array_training_path, targets_flattened_training)
    #np.save(predictions_per_sentence_training_path, predictions_sentence_training)
    #np.save(targets_per_sentence_training_path, targets_sentence_training)

    #print('\n3. REFORMATTING PREDICTIONS WITHOUT PADDINGS')
    #ocr_0_indeces = np.where(ocr_encodings_training_flattened == 0)[0]
    #predictions_no_pads_training = np.delete(predictions_flattened_training, ocr_0_indeces, axis=0)
    #targets_no_pads_training = np.delete(targets_flattened_training, ocr_0_indeces, axis=0)

    #np.save(predictions_no_pads_training_path, predictions_no_pads_training)
    #np.save(targets_no_pads_training_path, targets_no_pads_training)

    ##for ocr, prediction, target in zip(ocr_encodings_flattened, predictions, targets):
    ##    if ocr != 0:
    ##        predictions_no_pads.append(prediction.item())
    ##        targets_no_pads.append(target.item())

    ##with io.open(predictions_no_pads_path, mode='wb') as f_out:
    ##    pickle.dump(predictions_no_pads, f_out)
    ##if os.path.isfile(targets_no_pads_path):
    ##    with io.open(targets_no_pads_path, mode='wb') as f_out:
    ##        pickle.dump(targets_no_pads, f_out)

    #from sklearn.metrics import f1_score, precision_score, recall_score
    #f1_micro_training = round(f1_score(targets_flattened_training, predictions_flattened_training, average='micro'),3)
    #precision_micro_training = round(precision_score(targets_flattened_training, predictions_flattened_training, average='micro'),3)
    #recall_micro_training = round(recall_score(targets_flattened_training, predictions_flattened_training, average='micro'),3)

    #f1_weighted_training = round(f1_score(targets_flattened_training, predictions_flattened_training, average='weighted'),3)
    #precision_weighted_training = round(precision_score(targets_flattened_training, predictions_flattened_training, average='weighted'),3)
    #recall_weighted_training = round(recall_score(targets_flattened_training, predictions_flattened_training, average='weighted'),3)

    #print('\nMetrics per character (training):')
    #print('F1 (micro): {}\nPrecision (micro): {}\nRecall (micro): {}'.format(f1_micro_training, precision_micro_training, recall_micro_training))
    #print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted_training, precision_weighted_training, recall_weighted_training))

    #f1_weighted_sentence_training = round(f1_score(targets_sentence_training, predictions_sentence_training, average='weighted'),3)
    #precision_weighted_sentence_training = round(precision_score(targets_sentence_training, predictions_sentence_training, average='weighted'),3)
    #recall_weighted_sentence_training = round(recall_score(targets_sentence_training, predictions_sentence_training, average='weighted'),3)

    #print('\nMetrics per sentence (training):')
    #print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted_sentence_training, precision_weighted_sentence_training, recall_weighted_sentence_training))

    #f1_weighted_no_pads_training = round(f1_score(targets_no_pads_training, predictions_no_pads_training, average='weighted'),3)
    #precision_weighted_no_pads_training = round(precision_score(targets_no_pads_training, predictions_no_pads_training, average='weighted'),3)
    #recall_weighted_no_pads_training = round(recall_score(targets_no_pads_training, predictions_no_pads_training, average='weighted'),3)

    #print('\nMetrics per character (no pads) (training):')
    #print('F1 (weighted): {}\nPrecision (weighted): {}\nRecall (weighted): {}'.format(f1_weighted_no_pads_training, precision_weighted_no_pads_training, recall_weighted_no_pads_training))

################################################################################
@click.command()
@click.argument('loss-dir', type=click.Path(exists=True))
def evaluate_loss(loss_dir):
    '''
    \b
    Arguments:
    loss-dir --
    '''

    with io.open(loss_dir, mode='r') as f_in:
        losses = json.load(f_in)

    epoch_losses = {}

    for epoch_id, batch_losses in losses.items():
        if int(epoch_id) % 2 == 0:
            epoch_losses[epoch_id] = np.sum(batch_losses)

    sorted_epoch_losses = sorted(epoch_losses.items(), key=lambda kv: kv[1])

    sorted_epoch_losses = OrderedDict(sorted_epoch_losses)

################################################################################
@click.command()
@click.argument('align-dir', type=click.Path(exists=True))
@click.argument('pred-dir', type=click.Path(exists=True))
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
def evaluate_translator(align_dir, pred_dir, ocr_dir, gt_dir):
    '''
    \b
    Arguments:
    align-dir --
    pred-dir --
    ocr-dir --
    gt-dir --
    '''

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(alignments_path, size='total')

    with io.open(pred_sequences_path, mode='rb') as f_in:
        pred_sequences = pickle.load(f_in)
    with io.open(gt_sequences_path, mode='rb') as f_in:
        gt_sequences = pickle.load(f_in)
    with io.open(ocr_sequences_path, mode='rb') as f_in:
        ocr_sequences = pickle.load(f_in)

    size_dataset = find_max_mod(len(alignments), batch_size)

    alignments = alignments[0:size_dataset]

    print('\n1. COMPARISON WITH GT')

    ocr_cer = []
    for a in alignments:
        ocr_cer.append(a[5])

    pred_cer = []
    ocr_cer_filtered = []
    i = 0
    for pred, gt, o_cer in zip(pred_sequences, gt_sequences, ocr_cer):

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
        i+=1

    print('OCR CER: {}'.format(np.mean(ocr_cer_filtered)))
    print('Pred CER: {}'.format(np.mean(pred_cer)))

    print('\n2. FALSE CORRECTIONS RATIO')
    false_corrections_ratio = []
    for ocr, gt, pred in zip(ocr_sequences, gt_sequences, pred_sequences):
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

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('pred-dir', type=click.Path(exists=True))
def predict_detector(ocr_dir, gt_dir, targets_dir, model_dir, pred_dir):
    '''
    \b
    Arguments:
    ocr-dir --
    gt-dir --
    targets-dir --
    model-dir --
    pred-dir --
    '''

    in_dir, model_file = os.path.split(model_dir)
    model_file, model_ext = os.path.splitext(model_file)

    hyper_params_dir = os.path.join(in_dir, 'hyper_params'+model_file+'.json')

    print('\n1. LOAD DATA (ENCODINGS, ENCODING MAPPINGS)')

    ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    gt_encodings = np.load(gt_dir, allow_pickle=True)

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    ocr_encodings = ocr_encodings[0:size_dataset]
    gt_encodings = gt_encodings[0:size_dataset]

    assert ocr_encodings.shape == gt_encodings.shape

    #ocr_encodings_training = np.load(ocr_encodings_training_path, allow_pickle=True)#[:365000]#[:41200]
    #gt_encodings_training = np.load(gt_encodings_training_path, allow_pickle=True)#[:365000]#[:41200]
    #assert ocr_encodings_training.shape == gt_encodings_training.shape

    print('OCR encoding dimensions: {}'.format(ocr_encodings.shape))
    print('GT encoding dimensions: {}'.format(gt_encodings.shape))

    #print('OCR training encoding dimensions: {}'.format(ocr_encodings_training.shape))
    #print('GT training encoding dimensions: {}'.format(gt_encodings_training.shape))

    targets = np.load(targets_dir)
    targets = targets[0:size_dataset]
    print('Target dimensions: {}'.format(targets.shape))

    #targets_training = np.load(targets_training_path)
    #targets_training = targets_training#[0:365000]
    #print('Target training dimensions: {}'.format(targets_training.shape))

    print('\n2. INITIALIZE DATASET OBJECT')

    dataset = OCRCorrectionDataset(ocr_encodings, gt_encodings)
    #dataset_training = OCRCorrectionDataset(ocr_encodings_training, gt_encodings_training)

    print('Dataset size: {}'.format(len(dataset)))
    #print('Training size: {}'.format(len(dataset_training)))

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)

    input_size = hyper_params['input_size']
    hidden_size = hyper_params['hidden_size']
    output_size = hyper_params['output_size']
    batch_size = hyper_params['batch_size']
    seq_length = hyper_params['seq_length']
    n_layers = hyper_params['n_layers']
    dropout = hyper_params['dropout_prob']
    bidirectional = hyper_params['bidir']
    activation = hyper_params['activation']
    device = hyper_params['device']

    detector = DetectorLSTM(input_size, hidden_size, output_size, batch_size, n_layers, bidirectional=bidir, activation=activation, device=device).to(device)

    checkpoint = torch.load(model_dir, map_location=device)

    detector.load_state_dict(checkpoint['trained_detector']) # trained_detector

    detector.eval()

    print('\n4. PREDICT ERRORS')

    error_predictions = predict_iters_detector(dataset, targets, detector, batch_size, output_size, device=device)
    torch.save(error_predictions, pred_dir)

    #error_predictions_training = predict_iters_detector(dataset_training, targets_training, detector, batch_size, output_size, device=device)
    #torch.save(error_predictions_training, error_predictions_training_path)

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('hyper-params-dir', type=click.Path(exists=True))
@click.argument('code-to-token-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def predict_translator(ocr_dir, gt_dir, model_dir, hyper_params_dir,
                       code_to_token_dir, out_dir):
    '''
    \b
    Arguments:
    ocr-dir --
    gt-dir --
    model-dir --
    hyper-params-dir --
    code-to-token-dir --
    out-dir --
    '''

    ocr_sequences_dir = os.path.join(out_dir, 'ocr_sequences.npy')
    decoded_sequences_dir = os.path.join(out_dir, 'decoded_sequences.pkl')
    pred_sequences_dir = os.path.join(out_dir, 'pred_sequences.pkl')
    gt_sequences_dir = os.path.join(out_dir, 'gt_sequences.pkl')

    print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    ocr_encodings = np.load(ocr_encodings_path, allow_pickle=True)
    gt_encodings = np.load(gt_encodings_path, allow_pickle=True)

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    ocr_encodings = ocr_encodings[0:size_dataset]
    gt_encodings = gt_encodings[0:size_dataset]

    assert ocr_encodings.shape == gt_encodings.shape

    print('OCR encoding dimensions: {}'.format(ocr_encodings.shape))
    print('GT encoding dimensions: {}'.format(gt_encodings.shape))

    with io.open(code_to_token_dir, mode='r') as f_in:
        code_to_token_mapping = json.load(f_in)

    print('\n2. INITIALIZE DATASET OBJECT')

    dataset = OCRCorrectionDataset(ocr_encodings, gt_encodings)

    print('Testing size: {}'.format(len(dataset)))

    print('\n3. DEFINE HYPERPARAMETERS AND LOAD ENCODER/DECODER NETWORKS')

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)

    input_size = hyper_params['input_size']
    hidden_size = hyper_params['hidden_size']
    output_size = hyper_params['output_size']
    batch_size = hyper_params['batch_size']
    n_epochs = hyper_params['n_epochs']
    seq_length = hyper_params['seq_length']
    n_layers = hyper_params['n_layers']
    dropout = hyper_params['dropout_prob']
    with_attention = hyper_params['with_attention']
    teacher_forcing_ratio = hyper_params['teacher_forcing_ratio']
    device = torch.device(hyper_params['device'])

    encoder = EncoderLSTM(input_size, hidden_size, batch_size, num_layers, device=device)

    if with_attention:
        decoder = AttnDecoderLSTM(hidden_size, output_size, batch_size, seq_length, num_layers=num_layers, dropout=dropout, device=device)
    else:
        decoder = DecoderLSTM(hidden_size, output_size, batch_size, device=device)

    checkpoint = torch.load(model_dir, map_location=device)
    encoder.load_state_dict(checkpoint['trained_encoder'])
    decoder.load_state_dict(checkpoint['trained_decoder'])

    encoder.eval()
    decoder.eval()

    print('\n4. PREDICT SEQUENCES')

    decodings = predict_iters(dataset, encoder, decoder, batch_size, seq_length, with_attention, device=device)

    print('\n5. DECODE SEQUENCES')

    decoded_sequences = []
    pred_sequences = []
    for decoded_batch in decodings:
        for decoding in decoded_batch:
            decoded_sequence, joined_sequence = decode_sequence(list(decoding), code_to_token_mapping)
            decoded_sequences.append(decoded_sequence)
            pred_sequences.append(joined_sequence)

    gt_sequences = []
    for e in gt_encodings: # default: gt_encodings_subset
        decoded_sequence, joined_sequence = decode_sequence(list(e), code_to_token_mapping)
        gt_sequences.append(joined_sequence)

    ocr_sequences = []
    for o in ocr_encodings:
        ocr_sequence = []
        for e in o:
            if e == 0:
                break
            ocr_sequence.append(code_to_token_mapping[str(e)])
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

        ocr_sequences.append(''.join(ocr_sequence_filtered))

    with io.open(decoded_sequences_dir, mode='wb') as f_out:
        pickle.dump(decoded_sequences, f_out)
    with io.open(pred_sequences_dir, mode='wb') as f_out:
        pickle.dump(pred_sequences, f_out)
    with io.open(gt_sequences_dir, mode='wb') as f_out:
        pickle.dump(gt_sequences, f_out)
    with io.open(ocr_sequences_dir, mode='wb') as f_out:
        pickle.dump(ocr_sequences, f_out)

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('model-out-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True)) #only needed for encoding_size; maybe find alternative
@click.option('--hidden-size', default=512, help='Hidden dimension of RNN architecture.')
@click.option('--batch-size', default=200, help='The training batch size.')
@click.option('--n-epochs', default=1000, help='The number of training epochs.')
@click.option('--lr', default=0.0001, help='The learning rate.')
@click.option('--node-type', default='lstm', help='The RNN type (LSTM/GRU)')
@click.option('--n-layers', default=2, help='The number of RNN layers.')
@click.option('--bidir/--no-bidir', default=False, help='--bidir: Train model bidirectional; --no-bidir: Train model monodirectional.')
@click.option('--dropout-prob', default=0.2, help='The dropout probability.')
def train_detector(ocr_dir, gt_dir, targets_dir, model_out_dir, token_to_code_dir,
                   hidden_size, batch_size, n_epochs, lr, node_type, n_layers,
                   bidir, dropout_prob):
    '''
    Train detector component of OCR post-correction pipeline.

    \b
    Arguments:
    ocr-dir -- The absolute path to the OCR data
    gt-dir -- The absolute path to the GT data
    targets-dir -- The absolute path to the targets
    model-out-dir -- The absolute path for the trained models
    token-to-code-dir -- The absolute path to the token-encoding mapping
    '''

    out_dir, model_file = os.path.split(model_out_dir)
    model_file, model_ext = os.path.splitext(model_file)

    loss_dir = os.path.join(out_dir, 'losses_'+model_file+'.json')
    hyper_params_dir = os.path.join(out_dir, 'hyper_params'+model_file+'.json')

    print('\n1. LOAD DATA (ENCODINGS, ENCODING MAPPINGS)')

    ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    gt_encodings = np.load(gt_dir, allow_pickle=True)

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    ocr_encodings = ocr_encodings[0:size_dataset]
    gt_encodings = gt_encodings[0:size_dataset]

    assert ocr_encodings.shape == gt_encodings.shape

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)

    print('OCR encoding dimensions: {}'.format(ocr_encodings.shape))
    print('GT encoding dimensions: {}'.format(gt_encodings.shape))

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    encoding_size = len(token_to_code_mapping) + 1
    print('Token encodings: {}'.format(encoding_size))

    targets = np.load(targets_dir)
    targets = targets[0:size_dataset]
    print('Target dimensions: {}'.format(targets.shape))

    print('\n2. INITIALIZE DATASET OBJECT')

    training_set = OCRCorrectionDataset(ocr_encodings, gt_encodings)
    print('Training size : {}'.format(len(training_set)))

    print('\n3. DEFINE HYPERPARAMETERS AND INITIALIZE ENCODER/DECODER NETWORKS')

    input_size = encoding_size
    output_size = 3
    print('Input - Hidden - Output: {} - {} - {}'.format(input_size, hidden_size, output_size))
    seq_length = training_set[0].shape[-1]
    print('Sequence Length: {}'.format(seq_length))
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(n_epochs))
    print('Learning Rate: {}'.format(lr))
    print('RNN Node Type: {}'.format(node_type))
    print('RNN Layers: {}'.format(n_layers))
    if bidir:
        print('Bidirectional Model: {}'.format(True))
    else:
        print('Bidirectional Model: {}'.format(False))
    print('Dropout Probability: {}'.format(dropout_prob))
    activation = 'softmax'
    print('Activation Function: {}'.format(activation))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training Device: {}'.format(device))
    loss_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    print('Loss weights: {}'.format(loss_weights))

    hyper_params = {
        'input_size': input_size,
        'hidden_size:': hidden_size,
        'output_size': output_size,
        'seq_length': seq_length,
        'batch_size': batch_size,
        'bidir': bidir, # needs to be checked
        'node_type': node_type,
        'n_epochs': n_epochs,
        'learning_rate': lr,
        'n_layers': n_layers,
        'dropout_prob': dropout_prob,
    #    'loss_weights': loss_weights,
        'activation': activation,
        'device': device.type
    }

    with io.open(hyper_params_dir, mode='w') as params_file:
        json.dump(hyper_params, params_file)

    if node_type == 'lstm':
        detector = DetectorLSTM(input_size, hidden_size, output_size, batch_size, n_layers, bidirectional=bidir, activation=activation, device=device).to(device)
    elif node_type == 'gru':
        detector = DetectorGRU(input_size, hidden_size, output_size, batch_size, n_layers, activation=activation, device=device).to(device)


    print('\n4. TRAIN MODEL')
    trained_detector, optimizer \
        = train_iters_detector(model_out_dir, loss_dir, training_set, targets, detector, n_epochs=n_epochs,
                     batch_size=batch_size, learning_rate=lr, loss_weights=loss_weights, plot_every=5, print_every=1,
                     save_every=2, device=device)

    root, ext = os.path.splitext(model_out_dir)
    model_out_dir = root + '_final' + ext

    torch.save({
        'trained_detector': trained_detector.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, model_out_dir)

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('model-out-dir')
@click.argument('token-to-code-dir', type=click.Path(exists=True)) #only needed for encoding_size; maybe find alternative
@click.option('--approach', default='seq2seq', help='The OCR post-correction approach ("seq2seq" or "gan").')
@click.option('--hidden-size', default=512, help='Hidden dimension of RNN architecture. (default: 512)')
@click.option('--batch-size', default=200, help='The training batch size. (default: 200)')
@click.option('--n-epochs', default=1000, help='The number of training epochs. (default: 1000)')
@click.option('--lr', default=0.0001, help='The learning rate. (default: 0.0001)')
@click.option('--n-layers', default=2, help='The number of RNN layers. (default: 2)')
@click.option('--attention/--no-attention', default=True, help='--attention: Use attention mechanism; --no-attention: Use no attention mechanism. (default: True)')
@click.option('--dropout-prob', default=0.2, help='The dropout probability. (default: 0.2)')
@click.option('--teacher-ratio', default=0.5, help='The teacher ratio probability. (default: 0.5)')
def train_translator(ocr_dir, gt_dir, model_out_dir, token_to_code_dir,
                     approach, hidden_size, batch_size, n_epochs, lr, n_layers,
                     attention, dropout_prob, teacher_ratio):
    '''
    Train translator component of OCR post-correction pipeline.

    \b
    Arguments:
    ocr-dir -- The absolute path to the OCR data
    gt-dir -- The absolute path to the GT data
    model-out-dir -- The absolute path for the trained models
    token-to-code-dir -- The absolute path to the token-encoding mapping
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    model_out_dir = os.path.abspath(model_out_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)

    if not os.path.isdir(model_out_dir):
        os.mkdir(model_out_dir)

    today = date.today()
    model_name = 'translator_' + today.strftime("%d%m%y")

    model_dir = os.path.join(model_out_dir, model_name+'.pt')
    loss_dir = os.path.join(model_out_dir, 'losses_'+model_name+'.json')
    hyper_params_dir = os.path.join(model_out_dir, 'hyperparams_'+model_name+'.json')

    print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    gt_encodings = np.load(gt_dir, allow_pickle=True)

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    ocr_encodings = ocr_encodings[0:size_dataset]
    gt_encodings = gt_encodings[0:size_dataset]

    assert ocr_encodings.shape == gt_encodings.shape

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)

    import pdb; pdb.set_trace()

    print('OCR encoding dimensions: {}'.format(ocr_encodings.shape))
    print('GT encoding dimensions: {}'.format(gt_encodings.shape))

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    encoding_size = len(token_to_code_mapping) + 1
    print('Token encodings: {}'.format(encoding_size))

    print('\n2. INITIALIZE DATASET OBJECT')

    training_set = OCRCorrectionDataset(ocr_encodings, gt_encodings)
    len(training_set)
    print('Training size : {}'.format(len(training_set)))

    print('\n3. DEFINE HYPERPARAMETERS AND INITIALIZE ENCODER/DECODER NETWORKS')

    input_size = encoding_size
    output_size = input_size
    print('Input - Hidden - Output: {} - {} - {}'.format(input_size, hidden_size, output_size))
    seq_length = training_set[0].shape[-1]
    print('Sequence Length: {}'.format(seq_length))
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(n_epochs))
    print('Learning Rate: {}'.format(lr))
    print('RNN Layers: {}'.format(n_layers))
    print('Dropout Probability: {}'.format(dropout_prob))
    if attention:
        print('Learned with Attention: {}'.format(True))
    else:
        print('Learned with Attention: {}'.format(False))
    print('Teacher Forcing Ratio: {}'.format(teacher_ratio))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training Device: {}'.format(device))

    hyper_params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'seq_length': seq_length,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': lr,
        'num_layers': n_layers,
        'dropout': dropout_prob,
        'with_attention': attention,
        'teacher_forcing_ratio': teacher_ratio,
        'training_device': device.type
    }

    with io.open(hyper_params_dir, mode='w') as params_file:
        json.dump(hyper_params, params_file)

    if approach == 'seq2seq':
        encoder = EncoderLSTM(input_size, hidden_size, batch_size, n_layers, device=device).to(device)

        if attention:
            decoder = AttnDecoderLSTM(hidden_size, output_size, batch_size, seq_length, num_layers=n_layers, dropout=dropout_prob, device=device).to(device)
        else:
            decoder = DecoderLSTM(hidden_size, output_size, batch_size, device=device).to(device)

        print('\n4. TRAIN MODEL')
        trained_encoder, trained_decoder, encoder_optimizer, decoder_optimizer \
            = train_iters_seq2seq(model_dir, loss_dir, training_set,
                        encoder, decoder, n_epochs=n_epochs,
                        batch_size=batch_size, learning_rate=lr,
                        with_attention=attention, plot_every=5, print_every=1,
                        save_every=2, teacher_forcing_ratio=teacher_ratio,
                        device=device)

        root, ext = os.path.splitext(model_out_dir)
        model_final_out_dir = root + '_final' + ext

        torch.save({
            'trained_encoder': trained_encoder.state_dict(),
            'trained_decoder': trained_decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict()
        }, model_final_out_dir)

    elif approach == 'gan':
        generator = Generator(input_size, hidden_size, output_size, batch_size, n_layers, bidirectional=False, dropout=dropout_prob, activation='softmax', device=device).to(device)
        discriminator = Discriminator(input_size, hidden_size, output_size).to(device)

        trained_generator, trained_discriminator, generator_optimizer, \
            discriminator_optimizer = train_iters_gan(model_dir, loss_dir,
                training_set, generator, discriminator, n_epochs=n_epochs,
                batch_size=batch_size, learning_rate=lr, plot_every=5,
                print_every=1, save_every=2,
                teacher_forcing_ratio=teacher_ratio, device=device)

        root, ext = os.path.splitext(model_dir)
        model_dir = root + '_final' + ext

        torch.save({
            'trained_generator': trained_generator.state_dict(),
            'trained_discriminator': trained_discriminator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict()
        }, model_dir)
    else:
        print('NOTE: OCR Post-Correction approach must be either "seq2seq" or "gan".')
