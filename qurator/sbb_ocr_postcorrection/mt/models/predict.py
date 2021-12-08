import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict(input_tensor, trained_encoder, trained_decoder, seq_length, with_attention, device):
    '''
    Train one input sequence, ie. one forward pass.

    Keyword arguments:
    input_tensor (torch.Tensor) -- the input data
    trained_encoder -- the trained encoder network
    trained_decoder -- the trained decoder network
    sequence_length (int) -- the length of the sequence
    with_attention (bool) -- defines if attention network is applied
    device (str) -- the device used for training (cpu/cuda)

    Outputs:

    '''

    with torch.no_grad():

        ###################
        #                 #
        #  Encoding Step  #
        #                 #
        ###################

        encoder_hidden = trained_encoder.init_hidden_state()
        encoder_cell = trained_encoder.init_cell_state()

        # This needs to be checked; dimensions may be different
        input_length = input_tensor.shape[0]

        # Depends on structure of input_tensor
        batch_size = input_tensor.shape[1]

        encoder_outputs = torch.zeros(batch_size,
                                      input_length,
                                      trained_encoder.hidden_size,
                                      device=device)
        
        for ei in range(input_length):

            encoder_output, encoder_hidden, encoder_cell = trained_encoder(input_tensor[ei], encoder_hidden, encoder_cell)

            for bi in range(batch_size):
                encoder_outputs[bi, ei] += encoder_output[bi, 0]

        ###################
        #                 #
        #  Decoding Step  #
        #                 #
        ###################

        decoder_input = input_tensor[0].clone().detach()

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        #decoded_tokens = []
        decoded_tokens = np.zeros([batch_size, input_length], dtype=np.int64)
        topv_scores = torch.zeros([batch_size, input_length], dtype=torch.float64)
        #decoder_output_raw = torch.zeros((target_length, batch_size, trained_decoder.out.out_features), dtype=torch.float64)

        if with_attention:

            decoder_attentions = torch.zeros(seq_length, seq_length)
            #decoder_attentions = torch.zeros(batch_size, target_length)

            for di in range(seq_length):
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = trained_decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

                #decoder_output_raw[di, :] = decoder_output

                # attention array needs to be fixed
                #decoder_attentions[di] = decoder_attention.data

                topv, topi = decoder_output.data.topk(1)

                decoder_input = topi.squeeze().detach()

                decoded_tokens[:, di] = topi.squeeze(1)

                topv = topv.squeeze().detach()
                topv_scores[:, di] = topv
                #decoded_tokens.append(topi.item())
        else:

            for di in range(seq_length):
                decoder_output, decoder_hidden, decoder_cell = trained_decoder(decoder_input, decoder_hidden, decoder_cell)

               # decoder_output_raw[di, :] = decoder_output

                topv, topi = decoder_output.data.topk(1)

                decoder_input = topi.squeeze().detach()

                decoded_tokens[:, di] = topi.squeeze(1)
                #decoded_tokens.append(topi.item())

        #decoder_output_raw = decoder_output_raw.permute(1,0,2).numpy()

        return decoded_tokens, topv_scores#decoder_output_raw


def predict_detector(input_tensor, trained_detector, output_size, device):
    with torch.no_grad():

        ###################
        #                 #
        #  Encoding Step  #
        #                 #
        ###################

        detector_hidden = trained_detector.init_hidden_state()
        detector_cell = trained_detector.init_cell_state()

        # This needs to be checked; dimensions may be different
        input_length = input_tensor.shape[0]
        #target_length = target_tensor.shape[0]

        # Depends on structure of input_tensor
        batch_size = input_tensor.shape[1]

        #detector_outputs = torch.zeros(batch_size,
        #                              target_length,
        #                              output_size,
        #                              #trained_encoder.hidden_size,
        #                              device=device)
        detector_outputs = torch.zeros(input_length,
                                      batch_size,
                                      output_size,
                                      #trained_encoder.hidden_size,
                                      device=device)

        for di in range(input_length):

            #import pdb; pdb.set_trace()

            detector_output, lstm_output, detector_hidden, detector_cell = trained_detector(input_tensor[di], detector_hidden, detector_cell)

            detector_output_exp = torch.exp(detector_output)

            # needs to be checked!!!!
            #for bi in range(batch_size):
            #    detector_outputs[bi,di] = detector_output_exp[bi]
            detector_outputs[di] = detector_output_exp

            #import pdb; pdb.set_trace()
        ###################
        #                 #
        #  Decoding Step  #
        #                 #
        ###################

        return detector_outputs


def predict_iters(data_test, trained_encoder, trained_decoder, batch_size, seq_length, with_attention, device):
    '''

    '''

    batch_number = int(len(data_test) / batch_size)

    decodings = []

    for batch in DataLoader(data_test, batch_size=batch_size):
        # Tensor dimensions need to be checked
        input_tensor = batch[:, 0, :].to(device)
        input_tensor = torch.t(input_tensor)

        decoded_tokens, topv_scores = predict(input_tensor, trained_encoder, trained_decoder, seq_length, with_attention, device)     

        decodings.append(decoded_tokens)


    return decodings

def predict_iters_argmax(converter, testing_size, batch_size, seq_length, device):
    '''

    '''

    batch_number = int(testing_size / batch_size)
    predictions = []
    targets = []

    input_size = converter.fc1.in_features

    with torch.no_grad():

        for i in range(batch_number):

            input_tensor = torch.rand([batch_size, seq_length, input_size], device=device)
 
            #https://discuss.pytorch.org/t/set-max-value-to-1-others-to-0/44350/2
            max_values = input_tensor.argmax(2)
            target_tensor = nn.functional.one_hot(max_values, num_classes=input_size).float().to(device)
            targets.append(target_tensor)

            pred_tensor = converter(input_tensor)
            #pred_argmax = pred_tensor.argmax(2)
            predictions.append(pred_tensor)


    return predictions, targets

def predict_iters_detector(data_test, trained_detector, batch_size, output_size, device):
    '''

    '''
    batch_number = int(len(data_test)/batch_size)
    seq_length = data_test[0].shape[1]
    outputs = torch.zeros([batch_number, seq_length, batch_size, output_size])
    
    #target_index = 0
    batch_index = 0
    for batch in DataLoader(data_test, batch_size=batch_size):
        # Tensor dimensions need to be checked
        input_tensor = batch[:, 0, :].to(device)
        input_tensor = torch.t(input_tensor)
        #target_tensor = batch[:, 1, :].to(device)
        #target_tensor = torch.t(target_tensor)
        
        #target_tensor = torch.from_numpy(targets_test[target_index:target_index+batch_size]).to(device)
        #target_tensor = torch.t(target_tensor)
        #target_index += batch_size

        #import pdb; pdb.set_trace()

        detector_outputs = predict_detector(input_tensor, trained_detector, output_size, device)

        outputs[batch_index] = detector_outputs
        batch_index += 1

    return outputs
