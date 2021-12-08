from collections import OrderedDict
import click
from datetime import date
import io
import json
import numpy as np
import os
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch

from .models.error_detector import DetectorLSTM, DetectorGRU
from .models.gan import DiscriminatorCNN ,DiscriminatorLinear, DiscriminatorLSTM, GeneratorLSTM, GeneratorSeq2Seq
from .models.helper_models import ArgMaxConverter, ArgMaxConverterCNN
from .models.predict import predict, predict_detector, predict_iters, \
    predict_iters_detector, predict_iters_argmax
from .models.seq2seq import AttnDecoderLSTM, DecoderLSTM, EncoderLSTM
from .models.train import  train_iters_detector, train_iters_gan, \
    train_iters_argmax, train_iters_seq2seq

from qurator.sbb_ocr_postcorrection.data_structures import OCRCorrectionDataset
from qurator.sbb_ocr_postcorrection.feature_extraction.encoding import add_padding, encode_sequence, decode_sequence
from qurator.sbb_ocr_postcorrection.feature_extraction.tokenization import WordpieceTokenizer
from qurator.sbb_ocr_postcorrection.helpers import find_max_mod
from qurator.sbb_ocr_postcorrection.preprocessing.database import \
    load_alignments_from_sqlite, save_alignments_to_sqlite
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
    ocr-dir -- The path to the OCR data
    targets-dir -- The path to the targets
    targets-char-dir -- The path to the character target 
    error-pred-dir --
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    targets_dir = os.path.abspath(targets_dir)
    targets_char_dir = os.path.abspath(targets_char_dir)
    error_pred_dir = os.path.abspath(error_pred_dir)

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

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    targets_dir = os.path.abspath(targets_dir)
    pred_dir = os.path.abspath(pred_dir)
    gt_dir = os.path.abspath(gt_dir)
    hyper_params_dir = os.path.abspath(hyper_params_dir)
    pred_arr_dir = os.path.abspath(pred_arr_dir)
    pred_sent_dir = os.path.abspath(pred_sent_dir)
    pred_nopad_dir = os.path.abspath(pred_nopad_dir)
    targets_arr_dir = os.path.abspath(targets_arr_dir)
    targets_sent_dir = os.path.abspath(targets_sent_dir)
    targets_nopad_dir = os.path.abspath(targets_nopad_dir)

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
    loss-dir -- The path to the loss file
    '''

    # make paths absolute
    loss_dir = os.path.abspath(loss_dir)

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
@click.option('--batch-size', default=200, help='The training batch size.')
def evaluate_translator(align_dir, pred_dir, ocr_dir, gt_dir, batch_size):
    '''
    \b
    Arguments:
    align-dir -- The path to the aligned data
    pred-dir -- The path to the predicted, i.e. corrected, data
    ocr-dir -- The path to the OCR data
    gt-dir -- The path to the GT data
    '''

    # make paths absolute
    align_dir = os.path.abspath(align_dir)
    pred_dir = os.path.abspath(pred_dir)
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(align_dir, size='total')

    with io.open(pred_dir, mode='rb') as f_in:
        pred_sequences = pickle.load(f_in)
    with io.open(gt_dir, mode='rb') as f_in:
        gt_sequences = pickle.load(f_in)
    with io.open(ocr_dir, mode='rb') as f_in:
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

        p_cer = character_error_rate(aligned_pred, aligned_gt)

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
#@click.argument('ocr-dir', type=click.Path(exists=True))
#@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('hyper-params-dir', type=click.Path(exists=True))
#@click.argument('code-to-token-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.option('--testing-size', default=200, help='The testing size.')
def predict_argmax_converter(model_dir, hyper_params_dir, out_dir, testing_size):
    '''
    \b
    Arguments:
    model-dir --
    hyper-params-dir --
    out-dir --
    '''

    # make paths absolute
    #ocr_dir = os.path.abspath(ocr_dir)
    #gt_dir = os.path.abspath(gt_dir)
    model_dir = os.path.abspath(model_dir)
    hyper_params_dir = os.path.abspath(hyper_params_dir)
    #code_to_token_dir = os.path.abspath(code_to_token_dir)
    #out_dir = os.path.abspath(out_dir)

    predictions_dir = os.path.join(out_dir, 'predictions_argmax.pkl')
    targets_dir = os.path.join(out_dir, 'targets_argmax.pkl')
    #pred_sequences_dir = os.path.join(out_dir, 'pred_sequences.pkl')
    #gt_sequences_dir = os.path.join(out_dir, 'gt_sequences.pkl')

    #print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)

    #ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    #gt_encodings = np.load(gt_dir, allow_pickle=True)

    #size_dataset = find_max_mod(len(ocr_encodings), hyper_params['batch_size'])

    #ocr_encodings = ocr_encodings[0:size_dataset]
    #gt_encodings = gt_encodings[0:size_dataset]

    #assert ocr_encodings.shape == gt_encodings.shape

    #print('OCR encoding dimensions: {}'.format(ocr_encodings.shape))
    #print('GT encoding dimensions: {}'.format(gt_encodings.shape))

    #with io.open(code_to_token_dir, mode='r') as f_in:
    #    code_to_token_mapping = json.load(f_in)

    #print('\n2. INITIALIZE DATASET OBJECT')

    #dataset = OCRCorrectionDataset(ocr_encodings, gt_encodings)

    #print('Testing size: {}'.format(len(dataset)))

    #print('\n3. DEFINE HYPERPARAMETERS AND LOAD ENCODER/DECODER NETWORKS')

    input_size = hyper_params['input_size']
    hidden_size = hyper_params['hidden_size']
    output_size = hyper_params['output_size']
    batch_size = hyper_params['batch_size']
    n_epochs = hyper_params['n_epochs']
    seq_length = hyper_params['seq_length']
    num_layers = hyper_params['num_layers']
    dropout = hyper_params['dropout']
    device = torch.device(hyper_params['training_device'])

    converter = ArgMaxConverter(input_size, hidden_size, num_layers)

    checkpoint = torch.load(model_dir, map_location=device)
    converter.load_state_dict(checkpoint['trained_converter'])

    converter.eval()

    print('\n4. PREDICT SEQUENCES')

    predictions, targets = predict_iters_argmax(converter, testing_size, batch_size, seq_length, device=device)

    #import pdb; pdb.set_trace()


    with io.open(predictions_dir, mode='wb') as f_out:
        pickle.dump(predictions, f_out)
    with io.open(targets_dir, mode='wb') as f_out:
        pickle.dump(targets, f_out)

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('hyper-params-dir', type=click.Path(exists=True))
@click.argument('pred-dir', type=click.Path(exists=True))
def predict_detector(ocr_dir, gt_dir, targets_dir, model_dir, hyper_params_dir, pred_dir):
    '''
    \b
    Arguments:
    ocr-dir -- The path to the OCR data
    gt-dir -- The path to the GT data
    targets-dir -- The path to the targets
    model-dir -- The path to the trained model
    hyper-params-dir -- The path to the hyperparameter file 
    pred-dir -- The output path for the error predictions
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    targets_dir = os.path.abspath(targets_dir)
    model_dir = os.path.abspath(model_dir)
    hyper_params_dir = os.path.abspath(hyper_params_dir)
    pred_dir = os.path.abspath(pred_dir)

    in_dir, model_file = os.path.split(model_dir)
    model_file, model_ext = os.path.splitext(model_file)

    #hyper_params_dir = os.path.join(in_dir, 'hyper_params'+model_file+'.json')

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)
    
    batch_size = hyper_params['batch_size']
    
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

    input_size = hyper_params['input_size']
    hidden_size = hyper_params['hidden_size']
    output_size = hyper_params['output_size']
    batch_size = hyper_params['batch_size']
    seq_length = hyper_params['seq_length']
    n_layers = hyper_params['n_layers']
    dropout = hyper_params['dropout_prob']
    bidir = hyper_params['bidir']
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
    ocr-dir -- The path to the OCR data
    gt-dir -- The path to the GT data
    model-dir -- The path for the trained models
    hyper-params-dir -- The path to the hyperparameter file  
    code-to-token-dir -- The path to the encoding-token mapping 
    out-dir -- The path to the output directory
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    model_dir = os.path.abspath(model_dir)
    hyper_params_dir = os.path.abspath(hyper_params_dir)
    code_to_token_dir = os.path.abspath(code_to_token_dir)
    out_dir = os.path.abspath(out_dir)

    ocr_sequences_dir = os.path.join(out_dir, 'ocr_sequences.pkl')
    decoded_sequences_dir = os.path.join(out_dir, 'decoded_sequences.pkl')
    pred_sequences_dir = os.path.join(out_dir, 'pred_sequences.pkl')
    gt_sequences_dir = os.path.join(out_dir, 'gt_sequences.pkl')

    print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    with io.open(hyper_params_dir, mode='r') as f_in:
        hyper_params = json.load(f_in)

    ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    gt_encodings = np.load(gt_dir, allow_pickle=True)

    size_dataset = find_max_mod(len(ocr_encodings), hyper_params['batch_size'])

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

    encoder = EncoderLSTM(input_size, hidden_size, batch_size, n_layers, device=device)

    if with_attention:
        decoder = AttnDecoderLSTM(hidden_size, output_size, batch_size, seq_length, num_layers=n_layers, dropout=dropout, device=device)
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
@click.argument('detector-model-dir', type=click.Path(exists=True))
@click.argument('translator-model-dir', type=click.Path(exists=True))
@click.argument('hyper-params-detector-dir', type=click.Path(exists=True))
@click.argument('hyper-params-translator-dir', type=click.Path(exists=True))
@click.argument('code-to-token-detector-dir', type=click.Path(exists=True))
@click.argument('code-to-token-translator-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def run_two_step_pipeline_on_single_page(ocr_dir, 
        detector_model_dir, 
        translator_model_dir,
        hyper_params_detector_dir, 
        hyper_params_translator_dir,
        code_to_token_detector_dir,
        code_to_token_translator_dir,
        out_dir):
    '''
    ''' 

    with io.open(ocr_dir, mode='r') as f_in:
        ocr_data = json.load(f_in)

    with io.open(hyper_params_detector_dir, mode='r') as f_in:
        hyper_params_detector = json.load(f_in)
    with io.open(hyper_params_translator_dir, mode='r') as f_in:
        hyper_params_translator = json.load(f_in)

    with io.open(code_to_token_detector_dir, mode='r') as f_in:
        code_to_token_mapping_detector = json.load(f_in)
        token_to_code_mapping_detector = {token: code for code, token in code_to_token_mapping_detector.items()}
    with io.open(code_to_token_translator_dir, mode='r') as f_in:
        code_to_token_mapping_translator = json.load(f_in)
        token_to_code_mapping_translator = {token: code for code, token in code_to_token_mapping_translator.items()}

    def encode_features_for_single_page(data, token_to_code_mapping, seq_len):

        tok = WordpieceTokenizer(token_to_code_mapping, token_delimiter="<WSC>", unknown_char="<UNK>")
        print_examples=False
        pad_encoding=True

        if seq_len is None:
            ocr_encodings = []
            for i, line in enumerate(data['none']['P0001'][0]):
                tokenized_ocr = tok.tokenize(alignment[1], print_examples)
                ocr_encoding = encode_sequence(tokenized_ocr, token_to_code_mapping)
                ocr_encodings.append(ocr_encoding)
            #seq_len = find_longest_sequence(ocr_encodings, gt_encodings)
            #print('Max Length: {}'.format(seq_len))
        else:
            print('Max Length: {}'.format(seq_len))

        ocr_encodings = []

        for i, line in enumerate(data['none']['P0001'][0]):
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

        return ocr_encodings

    ocr_encodings = encode_features_for_single_page(ocr_data, token_to_code_mapping_detector, seq_len=40)

    batch_size = 200

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    # Fill last batch with empty lines (if last batch < batch_size)
    if len(ocr_encodings) > size_dataset:
        size_smallest_batch = len(ocr_encodings) - size_dataset 
        missing_lines_number = batch_size - size_smallest_batch
        zero_array = np.zeros([missing_lines_number, 40], dtype=int)

        ocr_encodings_batch_padded = np.concatenate((ocr_encodings, zero_array), axis=0) 
        size_dataset = ocr_encodings_batch_padded.shape[0]  
        
    #print('OCR testing encoding dimensions: {}'.format(ocr_encodings.shape))

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    detector_encoding_size = len(token_to_code_mapping_detector) + 1
    print('Token encodings: {}'.format(detector_encoding_size))

    #print('\n2. INITIALIZE DETECTOR DATASET OBJECT')

    #detector_dataset = OCRCorrectionDataset(ocr_encodings, gt_encodings)
    detector_dataset = OCRCorrectionDataset(ocr_encodings_batch_padded)

    print('Testing size: {}'.format(len(detector_dataset)))
    #print('Training size: {}'.format(len(dataset_training)))

    detector_input_size = detector_encoding_size
    detector_hidden_size = hyper_params_detector['hidden_size']
    detector_output_size = hyper_params_detector['output_size']
    detector_batch_size = hyper_params_detector['batch_size']
    seq_length = hyper_params_detector['seq_length']
    detector_num_layers = hyper_params_detector['n_layers']
    detector_dropout = hyper_params_detector['dropout_prob']
    detector_bidirectional = hyper_params_detector['bidir']
    detector_activation = hyper_params_detector['activation']
    detector_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = DetectorLSTM(detector_input_size, detector_hidden_size, detector_output_size, detector_batch_size, detector_num_layers, bidirectional=detector_bidirectional, activation=detector_activation, device=detector_device).to(detector_device)

    detector_checkpoint = torch.load(detector_model_dir, map_location=detector_device)

    detector.load_state_dict(detector_checkpoint['trained_detector']) # trained_detector

    detector.eval()

    print('\n4. PREDICT ERRORS')

    import pdb; pdb.set_trace()

    error_predictions = predict_iters_detector(detector_dataset, detector, detector_batch_size, detector_output_size, device=detector_device)

    #torch.save(error_predictions, error_predictions_dir)

    ##########

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
    predicted_labels_total = np.zeros((size_dataset, seq_length))

    batch_id = 0

    ############
    print('\n5. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')

    for predicted_batch in error_predictions:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1

        #target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch, threshold=0.99)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64).numpy()

        predicted_labels_total[target_index:(target_index+detector_batch_size), :] = batch_predicted_labels

        target_index += detector_batch_size

    seq_id_detector_pred_mapping = {} # needed to map corrected lines back to their original position
    
    predicted_sequence_labels = np.zeros((size_dataset-missing_lines_number, 1))
    predicted_labels_total = predicted_labels_total[:len(ocr_encodings), :]

    for seq_i, sequence in enumerate(predicted_labels_total):
        if 2 in sequence:
            predicted_sequence_labels[seq_i] = 1
            seq_id_detector_pred_mapping[str(seq_i)] = 1
        else:
            predicted_sequence_labels[seq_i] = 0
            seq_id_detector_pred_mapping[str(seq_i)] = 0

    ########
    print('\n CREATE TRANSLATOR SET')

    ocr_encodings_translator = encode_features_for_single_page(ocr_data, token_to_code_mapping_translator, seq_len=40)
    
    ocr_encodings_incorrect = []
    ocr_encodings_correct = []

    incorrect_id = 0
    correct_id = 0
    incorrect_line_mapping = {}
    correct_line_mapping = {}
    incorrect_lines = []
    for i in range(ocr_encodings_translator.shape[0]):
        if seq_id_detector_pred_mapping[str(i)] == 1:
            incorrect_line_mapping[str(i)] = str(incorrect_id)
            incorrect_id += 1
            correct_line_mapping[str(i)] = False 
            incorrect_lines.append(str(i))
            
            ocr_encodings_incorrect.append(ocr_encodings_translator[i])
        else:
            correct_line_mapping[str(i)] = str(correct_id)
            correct_id += 1
            incorrect_line_mapping[str(i)] = False
 
            ocr_encodings_correct.append(ocr_encodings_translator[i])
    
    ocr_encodings_correct = np.array(ocr_encodings_correct)
    ocr_encodings_incorrect = np.array(ocr_encodings_incorrect)

    size_dataset_translator = find_max_mod(len(ocr_encodings_incorrect), batch_size)

    # Fill last batch with empty lines (if last batch < batch_size)
    if len(ocr_encodings_incorrect) > size_dataset_translator:
        size_smallest_batch_translator = len(ocr_encodings_incorrect) - size_dataset_translator 
        missing_lines_number_translator = batch_size - size_smallest_batch_translator
        zero_array_translator = np.zeros([missing_lines_number_translator, 40], dtype=int)

        ocr_encodings_incorrect_batch_padded = np.concatenate((ocr_encodings_incorrect, zero_array_translator), axis=0) 
        size_dataset_translator = ocr_encodings_incorrect_batch_padded.shape[0] 
    
    import pdb; pdb.set_trace()

    ocr_lines = ocr_data['none']['P0001'][0]
    correct_sequences = []

    for i, out in seq_id_detector_pred_mapping.items():
        if out == 0:
            correct_sequences.append(ocr_lines[int(i)][1])
            
    import pdb; pdb.set_trace()

    
    #############################
    print('\n3.TRANSLATOR')
    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    translator_encoding_size = len(token_to_code_mapping_translator) + 1
    print('Token encodings: {}'.format(translator_encoding_size))

    print('\n3.1. INITIALIZE DATASET OBJECT')

    data_incorrect_size = ocr_encodings_incorrect.shape[0]

    #translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect_hack, gt_encodings_incorrect_hack)
    translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect_batch_padded)

    print('Testing size: {}'.format(len(translator_dataset_testing)))

    print('\n3.2. DEFINE HYPERPARAMETERS AND LOAD ENCODER/DECODER NETWORKS')

    translator_input_size = translator_encoding_size
    translator_hidden_size = hyper_params_translator['hidden_size']
    translator_output_size = translator_input_size
    translator_batch_size = hyper_params_translator['batch_size']
    translator_seq_length = hyper_params_translator['seq_length']
    translator_num_layers = hyper_params_translator['n_layers']
    translator_dropout = hyper_params_translator['dropout_prob']
    translator_with_attention = hyper_params_translator['with_attention']
    translator_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderLSTM(translator_input_size, translator_hidden_size, translator_batch_size, translator_num_layers, device=translator_device)

    if translator_with_attention:
        decoder = AttnDecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, translator_seq_length, num_layers=translator_num_layers, dropout=translator_dropout, device=translator_device)
    else:
        decoder = DecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, device=translator_device)

    translator_checkpoint = torch.load(translator_model_dir, map_location=translator_device)
    encoder.load_state_dict(translator_checkpoint['trained_encoder'])
    decoder.load_state_dict(translator_checkpoint['trained_decoder'])

    encoder.eval()
    decoder.eval()

    print('\n3.3. PREDICT SEQUENCES')

    translator_decodings = predict_iters(translator_dataset_testing, encoder, decoder, translator_batch_size, translator_seq_length, translator_with_attention, device=translator_device)

    print('\n3.4. DECODE SEQUENCES')

    translator_decoded_sequences = []
    translator_pred_sequences = []
    for decoded_batch in translator_decodings:
        for decoding in decoded_batch:
            decoded_sequence, joined_sequence = decode_sequence(list(decoding), code_to_token_mapping_translator)
            translator_decoded_sequences.append(decoded_sequence)
            translator_pred_sequences.append(joined_sequence)

    translator_pred_sequences = translator_pred_sequences[:len(ocr_encodings_incorrect)]

    #with io.open(translator_decoded_sequences_dir, mode='wb') as f_out:
    #    pickle.dump(translator_decoded_sequences, f_out)
    #with io.open(translator_pred_sequences_dir, mode='wb') as f_out:
    #    pickle.dump(translator_pred_sequences, f_out)

    # combine correct and corrected sequences; loop over whole data size
    corrected_data = []
    
    import pdb; pdb.set_trace()

    for i in range(len(ocr_lines)):
        if correct_line_mapping[str(i)]:
            corrected_data.append(correct_sequences[int(correct_line_mapping[str(i)])])
        if incorrect_line_mapping[str(i)]:
            corrected_data.append(translator_pred_sequences[int(incorrect_line_mapping[str(i)])])

    import pdb; pdb.set_trace()

    page_out_dir = os.path.join(out_dir, "corrected_page.txt")
    with io.open(page_out_dir, mode='w') as f_out:
        for line in corrected_data:
            f_out.write("%s\n" % line)
    
    import pdb; pdb.set_trace()
    incorrect_lines_id_out_dir = os.path.join(out_dir, "incorrect_lines_id.txt")
    with io.open(incorrect_lines_id_out_dir, mode='w') as f_out:
        for id in incorrect_lines:
            f_out.write("%s\n" % id)
    

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
#@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('aligned-dir', type=click.Path(exists=True))
@click.argument('detector-model-dir', type=click.Path(exists=True))
@click.argument('translator-model-dir', type=click.Path(exists=True))
#@click.argument('hyper-params-dir', type=click.Path(exists=True))
#@click.argument('code-to-token-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
def run_two_step_pipeline(ocr_dir, aligned_dir, detector_model_dir, 
    translator_model_dir, out_dir):
    '''
    \b
    Arguments:
    ocr-dir -- The path to the OCR data
    detector-model-dir -- The path for the trained detector model
    translator-model-dir -- The path for the trained translator model
    hyper-params-dir -- The path to the hyperparameter file  
    code-to-token-dir -- The path to the encoding-token mapping 
    out-dir -- The path to the output directory
    '''

    home_dir = '/home/robin'


    # path definitions
    #alignments_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_sliding_window_german_biased_2charges_170920.db'

    #ocr_encodings_testing_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_ocr_sliding_window_3_2charges_170920.npy'
    #gt_encodings_testing_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_gt_sliding_window_3_2charges_170920.npy'

    detector_token_to_code_path = home_dir + '/Qurator/used_data/features/dta/token_to_code_mapping_sliding_window_3_150620.json'     #token_to_code_mapping_3_2charges_110920.json'
    detector_code_to_token_path = home_dir + '/Qurator/used_data/features/dta/code_to_token_mapping_sliding_window_3_150620.json'     #code_to_token_mapping_3_2charges_110920.json'
    translator_token_to_code_path = home_dir + '/Qurator/used_data/features/dta/token_to_code_mapping_sliding_window_3_charge2_080920.json' #token_to_code_mapping_sliding_window_3_150620.json'
    translator_code_to_token_path = home_dir + '/Qurator/used_data/features/dta/code_to_token_mapping_sliding_window_3_charge2_080920.json'

    targets_testing_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_3_2charges_170920.npy'
    targets_testing_char_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_char_2charges_170920.npy'

    #detector_model_path = home_dir + '/Qurator/used_data/models/models_detector_sliding_window_512_3L_LSTM_bidirec_3_070920/trained_detector_model_512_3L_LSTM_bidirec_070920_138.pt'
    #translator_model_path = home_dir + '/Qurator/used_data/models/models_translator_sliding_window_256_1L_LSTM_monodirec_3_100920/trained_translator_model_256_1L_LSTM_monodirec_100920_876.pt'

    #error_predictions_testing_path = home_dir + '/Qurator/used_data/output_data/predictions_pipe_detector_testing_sliding_window_512_3L_LSTM_bidirec_3_170920_138.pt'

    #ocr_encodings_incorrect_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_ocr_pipe_sliding_window_3_2charges_170920.npy'
    #gt_encodings_incorrect_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_gt_pipe_sliding_window_3_2charges_170920.npy'
    #ocr_encodings_correct_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_ocr_pipe_sliding_window_3_2charges_170920.npy'
    #gt_encodings_correct_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_gt_pipe_sliding_window_3_2charges_170920.npy'

    #ocr_sequences_incorrect_path = home_dir + '/Qurator/used_data/output_data/sequences_incorrect_ocr_pipe_sliding_window_3_2charges_170920.pkl'
    #gt_sequences_incorrect_path = home_dir + '/Qurator/used_data/output_data/sequences_incorrect_gt_pipe_sliding_window_3_2charges_170920.pkl'
    #ocr_sequences_correct_path = home_dir + '/Qurator/used_data/output_data/sequences_correct_ocr_pipe_sliding_window_3_2charges_170920.pkl'
    #gt_sequences_correct_path = home_dir + '/Qurator/used_data/output_data/sequences_correct_gt_pipe_sliding_window_3_2charges_170920.pkl'

    ocr_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_incorrect_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_incorrect_gt_pipe_sliding_window_3_2charges_hack_170920.npy'
    ocr_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_ocr_pipe_sliding_window_3_2charges_hack_170920.npy'
    gt_sequences_correct_encoded_path = home_dir + '/Qurator/used_data/output_data/encoded_correct_gt_pipe_sliding_window_3_2charges_hack_170920.npy'

    #alignments_incorrect_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_incorrect_sliding_window_german_biased_2charges_170920.db'
    #alignments_correct_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_correct_sliding_window_german_biased_2charges_170920.db'


    #translator_ocr_sequences_path = home_dir + '/Qurator/used_data/features/dta/ocr_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.npy'
    #translator_decoded_sequences_path = home_dir + '/Qurator/used_data/output_data/decoded_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #decoded_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    #translator_pred_sequences_path = home_dir + '/Qurator/used_data/output_data/pred_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #pred_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'
    #translator_gt_sequences_path = home_dir + '/Qurator/used_data/output_data/gt_testing_sequences_pipe_translator_256_1L_LSTM_monodirec_2charges_170920_876.pkl' #gt_sequences_translator_256_1L_LSTM_monodirec_onestep_100920_970.pkl'


    print('\n1. LOAD DATA (ALIGNMENTS, ENCODINGS, ENCODING MAPPINGS)')

    error_predictions_dir = os.path.join(out_dir, 'error_predictions.pt')
    alignments_incorrect_dir = os.path.join(out_dir, 'alignments_incorrect.db')
    alignments_correct_dir = os.path.join(out_dir, 'alignments_correct.db')

    ocr_encodings_incorrect_dir = os.path.join(out_dir, 'encoded_ocr_incorrect.npy')
    #gt_encodings_incorrect_dir = os.path.join(out_dir, 'encoded_gt_incorrect.npy')
    ocr_encodings_correct_dir = os.path.join(out_dir, 'encoded_ocr_correct.npy')
    #gt_encodings_correct_dir = os.path.join(out_dir, 'encoded_gt_correct.npy')

    ocr_sequences_incorrect_dir = os.path.join(out_dir, 'sequences_ocr_incorrect.pkl')
    #gt_sequences_incorrect_dir = os.path.join(out_dir, 'sequences_gt_incorrect.pkl')
    ocr_sequences_correct_dir = os.path.join(out_dir, 'sequences_ocr_correct.pkl')
    #gt_sequences_correct_dir = os.path.join(out_dir, 'sequences_gt_correct.pkl')

    translator_ocr_sequences_dir = os.path.join(out_dir, 'translator_ocr_sequences.pkl')
    translator_decoded_sequences_dir = os.path.join(out_dir, 'translator_decoded_sequences.pkl')
    translator_pred_sequences_dir = os.path.join(out_dir, 'translator_pred_sequences.pkl')
    #translator_gt_sequences_dir = os.path.join(out_dir, 'translator_gt_sequences.pkl')

    detector_dir = os.path.split(detector_model_dir)[0]
    detector_hyper_params_dir = os.path.join(detector_dir, 'hyper_params_detector.json')
    translator_dir = os.path.split(translator_model_dir)[0]
    translator_hyper_params_dir = os.path.join(translator_dir, 'hyper_params_translator.json')

    with io.open(detector_hyper_params_dir, mode='r') as f_in:
        hyper_params_detector = json.load(f_in)
    with io.open(translator_hyper_params_dir, mode='r') as f_in:
        hyper_params_translator = json.load(f_in)

    ocr_encodings = np.load(ocr_dir, allow_pickle=True)
    #gt_encodings = np.load(gt_dir, allow_pickle=True)

    batch_size = hyper_params_detector['batch_size']

    size_dataset = find_max_mod(len(ocr_encodings), batch_size)

    size_dataset = 1000

    ocr_encodings = ocr_encodings[:size_dataset]
    #gt_encodings = gt_encodings[:size_dataset]

    #assert ocr_encodings.shape == gt_encodings.shape

    with io.open(detector_token_to_code_path, mode='r') as f_in:
        detector_token_to_code_mapping = json.load(f_in)
    with io.open(detector_code_to_token_path, mode='r') as f_in:
        detector_code_to_token_mapping = json.load(f_in)

    alignments, alignments_as_df, alignments_headers = load_alignments_from_sqlite(aligned_dir, size='total')

    alignments = alignments[:size_dataset]

    print('Alignments: {}'.format(len(alignments)))

    print('OCR testing encoding dimensions: {}'.format(ocr_encodings.shape))
    #print('GT testing encoding dimensions: {}'.format(gt_encodings.shape))

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    detector_encoding_size = len(detector_token_to_code_mapping) + 1
    print('Token encodings: {}'.format(detector_encoding_size))

    #detector_targets_testing = np.load(targets_testing_path)
    #detector_targets_testing = detector_targets_testing[0:size_dataset]
    #print('Target testing dimensions: {}'.format(detector_targets_testing.shape))

    detector_targets_char_testing = np.load(targets_testing_char_path)
    detector_targets_char_testing = detector_targets_char_testing[0:size_dataset]

    print('\n2. INITIALIZE DETECTOR DATASET OBJECT')

    #detector_dataset = OCRCorrectionDataset(ocr_encodings, gt_encodings)
    detector_dataset = OCRCorrectionDataset(ocr_encodings)


    print('Testing size: {}'.format(len(detector_dataset)))
    #print('Training size: {}'.format(len(dataset_training)))

    detector_input_size = detector_encoding_size
    detector_hidden_size = hyper_params_detector['hidden_size']
    detector_output_size = hyper_params_detector['output_size']
    detector_batch_size = hyper_params_detector['batch_size']
    seq_length = hyper_params_detector['seq_length']
    detector_num_layers = hyper_params_detector['n_layers']
    detector_dropout = hyper_params_detector['dropout_prob']
    detector_bidirectional = hyper_params_detector['bidir']
    detector_activation = hyper_params_detector['activation']
    detector_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = DetectorLSTM(detector_input_size, detector_hidden_size, detector_output_size, detector_batch_size, detector_num_layers, bidirectional=detector_bidirectional, activation=detector_activation, device=detector_device).to(detector_device)

    detector_checkpoint = torch.load(detector_model_dir, map_location=detector_device)

    detector.load_state_dict(detector_checkpoint['trained_detector']) # trained_detector

    detector.eval()

    print('\n4. PREDICT ERRORS')

    error_predictions = predict_iters_detector(detector_dataset, detector, detector_batch_size, detector_output_size, device=detector_device)

    torch.save(error_predictions, error_predictions_dir)
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
    predicted_labels_total = np.zeros((size_dataset, seq_length))

    batch_id = 0
    print('\n5. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')

    for predicted_batch in error_predictions:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1

        #target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch, threshold=0.99)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64).numpy()

        predicted_labels_total[target_index:(target_index+detector_batch_size), :] = batch_predicted_labels

        target_index += detector_batch_size

    predicted_sequence_labels = np.zeros((size_dataset, 1))
    for seq_i, sequence in enumerate(predicted_labels_total):
        if 2 in sequence:
            predicted_sequence_labels[seq_i] = 1
        else:
            predicted_sequence_labels[seq_i] = 0

    target_sequence_labels = np.zeros((len(detector_targets_char_testing), 1))
    for seq_i, sequence in enumerate(detector_targets_char_testing):
        if 2 in sequence:
            target_sequence_labels[seq_i] = 1
        else:
            target_sequence_labels[seq_i] = 0

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
    #gt_encodings_correct = []
    ocr_encodings_incorrect = []
    #gt_encodings_incorrect = []
    for alignment, sequence_label, ocr_encoding in zip(alignments, target_sequence_labels, ocr_encodings):# gt_encodings):
        if sequence_label == 1:
            ocr_encodings_incorrect.append(ocr_encoding)
    #        gt_encodings_incorrect.append(gt_encoding)
            alignments_incorrect.append(alignment)
        elif sequence_label == 0:
            ocr_encodings_correct.append(ocr_encoding)
    #        gt_encodings_correct.append(gt_encoding)
            alignments_correct.append(alignment)
    
    save_alignments_to_sqlite(alignments_incorrect, path=alignments_incorrect_dir, append=False)
    save_alignments_to_sqlite(alignments_correct, path=alignments_correct_dir, append=False)

    ocr_encodings_incorrect = np.array(ocr_encodings_incorrect)
    #gt_encodings_incorrect = np.array(gt_encodings_incorrect)
    ocr_encodings_correct = np.array(ocr_encodings_correct)
    #gt_encodings_correct = np.array(gt_encodings_correct)

    np.save(ocr_encodings_incorrect_dir, ocr_encodings_incorrect)
    #np.save(gt_encodings_incorrect_dir, gt_encodings_incorrect)
    np.save(ocr_encodings_correct_dir, ocr_encodings_correct)
    #np.save(gt_encodings_correct_dir, gt_encodings_correct)

    print('OCR encoding dimensions (incorrect): {}'.format(ocr_encodings_incorrect.shape))
    #print('GT encoding dimensions (incorrect): {}'.format(gt_encodings_incorrect.shape))

    # just placeholders for the moment
    gt_encodings_incorrect = None
    gt_encodings_correct = None

    encoding_arrays = [ocr_encodings_incorrect, gt_encodings_incorrect, ocr_encodings_correct, gt_encodings_correct]

    detector_output_sequences = [[], [], [], []]

    for i, encoding_array in enumerate(encoding_arrays):
        if encoding_array is not None:
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

    with io.open(ocr_sequences_incorrect_dir, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[0], f_out)
    #with io.open(gt_sequences_incorrect_dir, mode='wb') as f_out:
    #    pickle.dump(detector_output_sequences[1], f_out)
    with io.open(ocr_sequences_correct_dir, mode='wb') as f_out:
        pickle.dump(detector_output_sequences[2], f_out)
    #with io.open(gt_sequences_correct_dir, mode='wb') as f_out:
    #    pickle.dump(detector_output_sequences[3], f_out)
        
################################################################################

    #ocr_encodings_incorrect_hack = np.load(ocr_sequences_incorrect_encoded_path)
    #gt_encodings_incorrect_hack = np.load(gt_sequences_incorrect_encoded_path)
    #ocr_encodings_correct_hack = np.load(ocr_sequences_correct_encoded_path)
    #gt_encodings_correct_hack = np.load(gt_sequences_correct_encoded_path)
################################################################################

    print('\n3. TRANSLATOR MODEL')

    with io.open(translator_token_to_code_path, mode='r') as f_in:
        translator_token_to_code_mapping = json.load(f_in)
    with io.open(translator_code_to_token_path, mode='r') as f_in:
        translator_code_to_token_mapping = json.load(f_in)

################################################################################

    tok = WordpieceTokenizer(translator_token_to_code_mapping, token_delimiter="<WSC>", unknown_char="<UNK>")

    import pdb; pdb.set_trace()

    ocr_encodings_incorrect_hack = []

    for i, alignment in enumerate(detector_output_sequences[0]):

        tokenized_ocr = tok.tokenize(alignment, False)
        #tokenized_gt = tok.tokenize(alignment[4], False)

        ocr_encoding = encode_sequence(tokenized_ocr, translator_token_to_code_mapping)
        #gt_encoding = encode_sequence(tokenized_gt, translator_token_to_code_mapping)

        ocr_encodings_incorrect_hack.append(ocr_encoding)
        #translator_training_gt_encodings.append(gt_encoding)
    
    ocr_encodings_incorrect_hack = add_padding(ocr_encodings_incorrect_hack, seq_length)

    import pdb; pdb.set_trace()

################################################################################

    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    translator_encoding_size = len(translator_token_to_code_mapping) + 1
    print('Token encodings: {}'.format(translator_encoding_size))

    print('\n3.1. INITIALIZE DATASET OBJECT')

    data_incorrect_size = ocr_encodings_incorrect.shape[0]

    # define data size that can be cleanly divided batch_size
    if data_incorrect_size >= batch_size:
        data_incorrect_size -= (data_incorrect_size % batch_size)

    ocr_encodings_incorrect_hack = ocr_encodings_incorrect_hack[:data_incorrect_size]
    #gt_encodings_incorrect_hack = gt_encodings_incorrect_hack[:data_incorrect_size]

    #translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect_hack, gt_encodings_incorrect_hack)
    translator_dataset_testing = OCRCorrectionDataset(ocr_encodings_incorrect_hack)

    print('Testing size: {}'.format(len(translator_dataset_testing)))

    print('\n3.2. DEFINE HYPERPARAMETERS AND LOAD ENCODER/DECODER NETWORKS')

    translator_input_size = translator_encoding_size
    translator_hidden_size = hyper_params_translator['hidden_size']
    translator_output_size = translator_input_size
    translator_batch_size = hyper_params_translator['batch_size']
    translator_seq_length = hyper_params_translator['seq_length']
    translator_num_layers = hyper_params_translator['n_layers']
    translator_dropout = hyper_params_translator['dropout_prob']
    translator_with_attention = hyper_params_translator['with_attention']
    translator_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderLSTM(translator_input_size, translator_hidden_size, translator_batch_size, translator_num_layers, device=translator_device)

    if translator_with_attention:
        decoder = AttnDecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, translator_seq_length, num_layers=translator_num_layers, dropout=translator_dropout, device=translator_device)
    else:
        decoder = DecoderLSTM(translator_hidden_size, translator_output_size, translator_batch_size, device=translator_device)

    translator_checkpoint = torch.load(translator_model_dir, map_location=translator_device)
    encoder.load_state_dict(translator_checkpoint['trained_encoder'])
    decoder.load_state_dict(translator_checkpoint['trained_decoder'])

    encoder.eval()
    decoder.eval()

    print('\n3.3. PREDICT SEQUENCES')

    translator_decodings = predict_iters(translator_dataset_testing, encoder, decoder, translator_batch_size, translator_seq_length, translator_with_attention, device=translator_device)

    print('\n3.4. DECODE SEQUENCES')

    translator_decoded_sequences = []
    translator_pred_sequences = []
    for decoded_batch in translator_decodings:
        for decoding in decoded_batch:
            decoded_sequence, joined_sequence = decode_sequence(list(decoding), translator_code_to_token_mapping)
            translator_decoded_sequences.append(decoded_sequence)
            translator_pred_sequences.append(joined_sequence)

    with io.open(translator_decoded_sequences_dir, mode='wb') as f_out:
        pickle.dump(translator_decoded_sequences, f_out)
    with io.open(translator_pred_sequences_dir, mode='wb') as f_out:
        pickle.dump(translator_pred_sequences, f_out)

    #translator_gt_sequences = []
    #for e in gt_encodings_incorrect: # default: gt_encodings_subset
    #    decoded_sequence, joined_sequence = decode_sequence(list(e), translator_code_to_token_mapping)
    #    translator_gt_sequences.append(joined_sequence)

    #with io.open(translator_gt_sequences_dir, mode='wb') as f_out:
    #    pickle.dump(translator_gt_sequences, f_out)

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

    with io.open(translator_ocr_sequences_dir, mode='wb') as f_out:
        pickle.dump(translator_ocr_sequences, f_out)
################################################################################
#
# The part below compares corrected OCR with GT; not needed for correction
# 
################################################################################
    #print('\n3.5. COMPARISON WITH GT')

    #ocr_cer = []
    #for a in alignments:
    #    ocr_cer.append(a[5])

    #pred_cer = []
    #ocr_cer_filtered = []
    #i = 0
    #for pred, gt, o_cer in zip(translator_pred_sequences, translator_gt_sequences, ocr_cer):

    #    #if not (o_cer >= 0.0 and o_cer < 0.02):
    #    #    continue

    #    aligned_sequence = seq_align(pred, gt)
    #    aligned_pred = []
    #    aligned_gt = []
    #    for alignment in aligned_sequence:
    #        if alignment[0] == None:
    #            aligned_pred.append(' ')
    #        else:
    #            aligned_pred.append(alignment[0])
    #        if alignment[1] == None:
    #            aligned_gt.append(' ')
    #        else:
    #            aligned_gt.append(alignment[1])
    #    aligned_pred = ''.join(aligned_pred)
    #    aligned_gt = ''.join(aligned_gt)
    #    assert len(aligned_pred) == len(aligned_gt)

    #    p_cer = character_error_rate(aligned_pred, aligned_gt)

    #    pred_cer.append(p_cer)
    #    ocr_cer_filtered.append(o_cer)

    #    if (i+1) % 10000 == 0:
    #        print(i+1)
    #    i += 1

    #print('OCR CER: {}'.format(np.mean(ocr_cer_filtered)))
    #print('Pred CER: {}'.format(np.mean(pred_cer)))

    #print('\n3.6. FALSE CORRECTIONS RATIO')
    #false_corrections_ratio = []
    #for ocr, gt, pred in zip(translator_ocr_sequences, translator_gt_sequences, translator_pred_sequences):
    #    if not len(ocr) == len(gt) == len(pred):
    #        max_length = max(len(ocr), len(gt), len(pred))

    #        if not (len(ocr) - max_length) == 0:
    #            ocr += (abs((len(ocr) - max_length)) * ' ')
    #        if not (len(gt) - max_length) == 0:
    #            gt += (abs((len(gt) - max_length)) * ' ')
    #        if not (len(pred) - max_length) == 0:
    #            pred += (abs((len(pred) - max_length)) * ' ')

    #        assert len(ocr) == len(gt) == len(pred)

    #    false_corrections_count = 0
    #    for o, g, p in zip(ocr, gt, pred):
    #        if (o == g) and p != g:
    #            false_corrections_count += 1
    #    false_corrections_ratio.append(false_corrections_count/len(pred))

    #print('False corrections ratio: {}'.format(np.mean(false_corrections_ratio)))

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('targets-dir', type=click.Path(exists=True))
@click.argument('model-out-dir', type=click.Path(exists=True))
@click.argument('token-to-code-dir', type=click.Path(exists=True)) #only needed for encoding_size; maybe find alternative
@click.option('--hidden-size', default=512, help='Hidden dimension of RNN architecture. (default: 512)')
@click.option('--batch-size', default=200, help='The training batch size. (default: 200)')
@click.option('--n-epochs', default=1000, help='The number of training epochs. (default: 1000)')
@click.option('--lr', default=0.0001, help='The learning rate. (default: 0.0001)')
@click.option('--node-type', default='lstm', help='The RNN type (LSTM/GRU). (default: lstm)')
@click.option('--n-layers', default=2, help='The number of RNN layers. (default: 2)')
@click.option('--bidir/--no-bidir', default=False, help='--bidir: Train model bidirectional; --no-bidir: Train model monodirectional. (default: false)')
@click.option('--dropout-prob', default=0.2, help='The dropout probability. (default: 0.2)')
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

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    targets_dir = os.path.abspath(targets_dir)
    model_out_dir = os.path.abspath(model_out_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)

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
        'hidden_size': hidden_size,
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
#@click.argument('one-hot-dir', type=click.Path(exists=True))
#@click.argument('target-dir', type=click.Path(exists=True))
@click.argument('model-out-dir')
@click.argument('token-to-code-dir', type=click.Path(exists=True)) #only needed for encoding_size; maybe find alternative
@click.option('--approach', default='linear', help='Argmax conversion approach: "linear" or "cnn". (default: "linear")')
@click.option('--hidden-size', default=512, help='Hidden dimension of RNN architecture. (default: 512)')
@click.option('--seq-length', default=40, help='The sequence length. (default: 40)')
@click.option('--training-size', default=100000, help='The training size. (default: 100000)')
@click.option('--batch-size', default=200, help='The training batch size. (default: 200)')
@click.option('--n-epochs', default=1000, help='The number of training epochs. (default: 1000)')
@click.option('--lr', default=0.0001, help='The learning rate. (default: 0.0001)')
@click.option('--n-layers', default=2, help='The number of RNN layers. (default: 2)')
@click.option('--dropout-prob', default=0.2, help='The dropout probability. (default: 0.2)')
def train_argmax_converter(model_out_dir, token_to_code_dir, approach, hidden_size, 
                    seq_length, training_size, batch_size, n_epochs, lr, 
                    n_layers, dropout_prob):
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
    model_out_dir = os.path.abspath(model_out_dir)
    token_to_code_dir = os.path.abspath(token_to_code_dir)

    if not os.path.isdir(model_out_dir):
        os.mkdir(model_out_dir)

    today = date.today()
    model_name = 'argmax_converter_' + today.strftime("%d%m%y")

    model_dir = os.path.join(model_out_dir, model_name+'.pt')
    loss_dir = os.path.join(model_out_dir, 'losses_'+model_name+'.json')
    hyper_params_dir = os.path.join(model_out_dir, 'hyperparams_'+model_name+'.json')

    size_dataset = find_max_mod(training_size, batch_size)

    with io.open(token_to_code_dir, mode='r') as f_in:
        token_to_code_mapping = json.load(f_in)
    
    # add 1 for additional 0 padding, i.e. padded 0 are treated as vocab
    encoding_size = len(token_to_code_mapping) + 1

    print('\n1. DEFINE HYPERPARAMETERS AND INITIALIZE CONVERTER NETWORK')

    input_size = encoding_size
    output_size = input_size
    print('Input - Hidden - Output: {} - {} - {}'.format(input_size, hidden_size, output_size))
    print('Sequence Length: {}'.format(seq_length))
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(n_epochs))
    print('Learning Rate: {}'.format(lr))
    print('Dropout Probability: {}'.format(dropout_prob))
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
        'training_device': device.type
    }

    with io.open(hyper_params_dir, mode='w') as params_file:
        json.dump(hyper_params, params_file)
    
    if approach == 'linear':
        converter = ArgMaxConverter(input_size, hidden_size, n_layers).to(device)
    elif approach == 'cnn':
        converter = ArgMaxConverterCNN(input_size, hidden_size)

    print('\n2. TRAIN MODEL')
    trained_converter, converter_optimizer = train_iters_argmax(model_dir, 
            loss_dir, training_size, converter, n_epochs=n_epochs, 
            seq_length=seq_length, batch_size=batch_size, learning_rate=lr, 
            print_every=1, plot_every=5, save_every=2, device=device)

    root, ext = os.path.splitext(model_dir)
    model_final_out_dir = root + '_final' + ext

    torch.save({
        'trained_converter': trained_converter.state_dict(),
        'converter_optimizer': converter_optimizer.state_dict(),
    }, model_final_out_dir)


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
    
    code_to_token_mapping = {code: token for token, code in token_to_code_mapping.items()}

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
    output_size_discriminator = 2
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
        
        #just for testing
        #output_size = hidden_size

        generator = GeneratorSeq2Seq(input_size=input_size, hidden_size=hidden_size, output_size=output_size, batch_size=batch_size, seq_length=seq_length, rnn_type='lstm',
                        n_layers=n_layers, bidirectional=False, dropout=dropout_prob, activation='softmax', device=device).to(device)
        
        #discriminator = DiscriminatorLSTM(input_size, hidden_size, output_size_discriminator, batch_size, device=device).to(device)

        discriminator = DiscriminatorCNN(input_size=input_size, hidden_size=hidden_size, kernel_size=2, stride=2, padding=1, lrelu_neg_slope=0.2, dropout_prob=0.5).to(device)

        print_gradient = True

        trained_generator, trained_discriminator, generator_optimizer, \
            discriminator_optimizer = train_iters_gan(model_dir, loss_dir,
                training_set, generator, discriminator, n_epochs=n_epochs,
                batch_size=batch_size, learning_rate=lr, 
                code_to_token_mapping=code_to_token_mapping, plot_every=5,
                print_every=1, save_every=2, print_gradient=print_gradient, 
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
