import io
import os
import numpy as np
import pickle
import torch
import sys
home_dir = os.path.expanduser('~')

if __name__ == '__main__':

    ocr_encodings_validation_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_ocr_sliding_window_3_charge1_160920.npy'#encoded_validation_ocr_3_charge1_110920.npy'
    ocr_encodings_training_path = home_dir + '/Qurator/used_data/features/dta/encoded_training_ocr_sliding_window_3_charge1_160920.npy'#encoded_training_ocr_3_charge1_110920.npy'
    targets_validation_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_3_charge1_160920.npy'#detector_target_validation_german_3_charge1_110920.npy'
    targets_training_path = home_dir + '/Qurator/used_data/features/dta/detector_target_training_sliding_window_german_3_charge1_160920.npy'
    error_predictions_validation_path = home_dir + '/Qurator/used_data/output_data/predictions_detector_testing_sliding_window_512_3L_LSTM_bidirec_3_160920_138.pt'
    #error_predictions_training_path = home_dir + '/Qurator/used_data/output_data/predictions_detector_training_512_3L_1_210820_436.pt'

    targets_validation_char_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_char_charge1_160920.npy'#detector_target_testing_char_00-10_sliding_window_german_softmax_070820.npy'


    print('\n1. LOAD DATA (TARGETS, ERROR PREDICTIONS)')

    targets_validation = np.load(targets_validation_path)
    targets_validation = targets_validation[0:56800]#[0:20600]
    print('Validation target dimensions: {}'.format(targets_validation.shape))

    targets_training = np.load(targets_training_path)
    targets_training = targets_training[0:365000]
    print('Trainig target dimensions: {}'.format(targets_training.shape))

    targets_char_validation = np.load(targets_validation_char_path)
    targets_char_validation = targets_char_validation[0:56800]#[0:20600]

    error_predictions_validation = torch.load(error_predictions_validation_path)
    print('Error prediction validation dimensions: {}'.format(error_predictions_validation.shape))
    #error_predictions_training = torch.load(error_predictions_training_path)
    #print('Error prediction training dimensions: {}'.format(error_predictions_training.shape))

    ocr_encodings_validation = np.load(ocr_encodings_validation_path, allow_pickle=True)[0:56800]#[:20600]
    print('OCR validation encodings dimensions: {}'.format(ocr_encodings_validation.shape))
    ocr_encodings_validation_flattened = ocr_encodings_validation.reshape([-1])

    ocr_encodings_training = np.load(ocr_encodings_training_path, allow_pickle=True)[:365000]
    print('OCR training encodings dimensions: {}'.format(ocr_encodings_training.shape))
    ocr_encodings_training_flattened = ocr_encodings_training.reshape([-1])


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
    predicted_labels_total = np.zeros_like(targets_validation)

    batch_id = 0
    print('\n2. REFORMATTING TOTAL PREDICTIONS AND SENTENCE-WISE:')
    for predicted_batch in error_predictions_validation:
        #print('Batch ID: {}'.format(batch_id))
        batch_id += 1

        #target_tensor = torch.from_numpy(targets_testing[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)

        batch_predicted_labels = convert_softmax_prob_to_label(predicted_batch, threshold=0.99)
        batch_predicted_labels = torch.t(batch_predicted_labels).type(torch.int64).numpy()

        predicted_labels_total[target_index:(target_index+batch_size), :] = batch_predicted_labels

        target_index += batch_size

    predicted_sequence_labels = np.zeros((len(targets_validation), 1))
    for seq_i, sequence in enumerate(predicted_labels_total):
        if 2 in sequence:
            predicted_sequence_labels[seq_i] = 1
        else:
            predicted_sequence_labels[seq_i] = 0

    #import pdb; pdb.set_trace()

    target_sequence_labels = np.zeros((len(targets_char_validation), 1))
    for seq_i, sequence in enumerate(targets_char_validation):
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
