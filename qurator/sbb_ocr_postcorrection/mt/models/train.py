import io
import json
import os
import random
import time
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from .gan import fake_loss, real_loss, Embedder
from qurator.sbb_ocr_postcorrection.feature_extraction.encoding import decode_sequence
from qurator.sbb_ocr_postcorrection.helpers import timeSince, showPlot

def train_detector(input_tensor, target_tensor, detector, optimizer,
          criterion, device):
    '''
    Train one input sequence, ie. one forward pass.

    Keyword arguments:
    input_tensor (torch.Tensor) -- the input data
    target_tensor (torch.Tensor) -- the target data
    detector -- the detector network
    optimizer -- the optimization algorithm
    criterion -- the loss function
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training (cpu/cuda)

    Outputs:
    the loss averaged by target length (float)
    '''

    ###################
    #                 #
    #  Encoding Step  #
    #                 #
    ###################

    detector_hidden = detector.init_hidden_state()
    if detector.node_type == 'lstm':
        detector_cell = detector.init_cell_state()

    optimizer.zero_grad()

    # This needs to be checked; dimensions may be different
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    # Depends on structure of input_tensor
    batch_size = input_tensor.shape[1]

    # Batch size implementation needs to be checked
    #detector_outputs = torch.zeros(batch_size,
    #                              target_length,
    #                              #encoder.hidden_size,
    #                              device=device)

    loss = 0

    if detector.node_type == 'lstm':
        for di in range(input_length):

            detector_output, lstm_output, detector_hidden, detector_cell = \
                detector(input_tensor[di], detector_hidden, detector_cell)

            loss += criterion(detector_output, target_tensor[di])

    elif detector.node_type == 'gru':
        for di in range(input_length):

            detector_output, gru_output, detector_hidden = \
                detector(input_tensor[di], detector_hidden)

            loss += criterion(detector_output, target_tensor[di])

    ###############################
    #                             #
    #  Backprop and optimization  #
    #                             #
    ###############################

    loss.backward()

    optimizer.step()

    return loss.item() / target_length

def train_iters_detector(model_path, loss_path, data_train, targets_train,
               detector, n_epochs, batch_size, learning_rate, loss_weights,
               print_every=5, plot_every=20, save_every=2, device='cpu'):
    '''
    Run train iteration.

    Keyword arguments:
    data_train (Custom PyTorch Dataset) -- the training data
    detector -- the encoder network
    n_epochs (int) -- the number of training epochs
    batch_size (int) -- the batch size
    learning_rate (float) -- the learning rate
    print_every (int) -- defines print status intervall
    plot_every (int) -- defines plotting intervall
    device (str) -- the device used for training

    Outputs:
    detector -- the trained detector
    '''
    start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.AdamW(detector.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(weight=loss_weights)
    #criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()

    loss_dict = {}

    for epoch in range(1, n_epochs + 1):

        loss_list = []
        target_index = 0

        for batch in DataLoader(data_train, batch_size=batch_size):

            # Tensor dimensions need to be checked
            input_tensor = batch[:, 0, :].to(device)
            input_tensor = torch.t(input_tensor)
            #target_tensor = batch[:, 1, :].to(device)
            #target_tensor = torch.t(target_tensor)
            target_tensor = torch.from_numpy(targets_train[target_index:target_index+batch_size]).to(device)
            target_tensor = torch.t(target_tensor)
            target_index += batch_size

            loss = train_detector(input_tensor,
                         target_tensor,
                         detector,
                         optimizer,
                         criterion,
                         device)

            loss_list.append(loss)

            print_loss_total += loss
            plot_loss_total += loss

        loss_dict[epoch] = loss_list

        if epoch % save_every == 0:
            root, ext = os.path.splitext(model_path)
            epoch_path = root + '_' + str(epoch) + ext
            torch.save({
                'trained_detector': detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, epoch_path)
            with io.open(loss_path, mode='w') as loss_file:
                json.dump(loss_dict, loss_file)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('{:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                    epoch,
                                                    int(epoch/n_epochs*100),
                                                    print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

    return detector, optimizer

def train_gan(input_tensor, target_tensor, generator, discriminator,
          generator_optimizer, discriminator_optimizer, criterion,
          teacher_forcing_ratio, code_to_token_mapping, device, embedder):
    '''
    Train one input sequence, ie. one forward pass.

    Keyword arguments:
    input_tensor (torch.Tensor) -- the input data
    target_tensor (torch.Tensor) -- the target data
    encoder -- the encoder network
    decoder -- the decoder network
    encoder_optimizer -- the encoder optimization algorithm
    decoder_optimizer -- the decoder optimization algorithm
    criterion -- the loss function
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training (cpu/gpu)

    Outputs:
    the loss averaged by target length (float)
    '''

    #generator_hidden = generator.encoder.init_hidden_state()
    #generator_cell = generator.encoder.init_cell_state()

    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    # This needs to be checked; dimensions may be different
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    # Depends on structure of input_tensor
    batch_size = input_tensor.shape[1]

    # Batch size implementation needs to be checked
    #generator_outputs = torch.zeros(batch_size,
     #                             target_length,
      #                            generator.output_size,
       #                           device=device)

    #generated_tensor_g = torch.zeros([batch_size, target_length], dtype=torch.int64, device=device)

    use_teacher_forcing = True if \
        random.random() < teacher_forcing_ratio else False

    
    #input_tensor = embedder(input_tensor)
    #target_tensor = embedder(target_tensor)

    ########################################
    #                                      #
    #  Generate (Translate) OCR Sequences  #
    #                                      #
    ########################################

    #########################
    #                       #
    #  Train Discriminator  #
    #                       #
    #########################

    import pdb; pdb.set_trace()

    #target_tensor = torch.t(target_tensor)
    target_tensor = nn.functional.one_hot(target_tensor)

    discriminator_output_real = discriminator(target_tensor)
    d_real_loss = real_loss(discriminator_output_real, criterion, smooth=False, device=device)
    
    #import pdb; pdb.set_trace()

    #target_tensor = torch.t(target_tensor)

    generated_tensor_d = generator(input_tensor, target_tensor, input_length,
                            target_length, use_teacher_forcing)
    
    import pdb; pdb.set_trace()


    #if generated_tensor_d.shape != target_tensor.shape:
    #    generated_tensor_d = generated_tensor_d.permute(1,0,2)
    
    
    #generated_tensor_d = generated_tensor_d.transpose(0,1)
    
    #decoded_generated_tensor_d = torch.zeros([200, 40], requires_grad=True)

    import pdb; pdb.set_trace()

    #for i in range(decoded_generated_tensor_d.shape[0]):
    #    topv, topi = generated_tensor_d[i].data.topk(1)

    #    decoded_generated_tensor_d[i, :] = topi.squeeze()  

    #import pdb; pdb.set_trace()

    #generated_tensor_d = generated_tensor_d.type(torch.LongTensor)

    #decoded_generated_tensor_d = decoded_generated_tensor_d.type(torch.LongTensor)


    #generated_tensor_d = decoded_generated_tensor_d.to(device)

    # 1. Train with target data (GT)

    #d_real_loss = 0
    #d_fake_loss = 0

    #import pdb; pdb.set_trace()

    discriminator_output_fake = discriminator(generated_tensor_d)
    d_fake_loss = fake_loss(discriminator_output_fake, criterion, device=device)

    #import pdb; pdb.set_trace()

    d_loss = d_real_loss + d_fake_loss

    ###################
    #                 #
    # Train Generator #
    #                 #
    ###################

    #target_tensor = torch.t(target_tensor)

    generated_tensor_g = generator(input_tensor, target_tensor, input_length,
                            target_length, use_teacher_forcing)

    
    #generated_tensor_g = generated_tensor_g.transpose(0,1)
    
    #decoded_generated_tensor_g = torch.zeros(200, 40, requires_grad=True)

    #for i in range(decoded_generated_tensor_g.shape[0]):
    #    topv, topi = generated_tensor_g[i].data.topk(1)

    #    decoded_generated_tensor_g[i, :] = topi.squeeze() 

    import pdb; pdb.set_trace() 

    #generated_tensor_g = generated_tensor_g.type(torch.LongTensor)

    #decoded_generated_tensor_g = decoded_generated_tensor_g.type(torch.LongTensor)

    #generated_tensor_g = decoded_generated_tensor_g.to(device)

    #generated_tensor_g = torch.t(generated_tensor_g)

    g_loss = 0

    #target_tensor = torch.t(target_tensor)

    discriminator_output_fake_2 = discriminator(generated_tensor_g)
    g_loss = real_loss(discriminator_output_fake_2, criterion, smooth=False, device=device)


    ###############################
    #                             #
    #  Backprop and optimization  #
    #                             #
    ###############################

    g_loss.backward(retain_graph=True)
    d_loss.backward()

    import pdb; pdb.set_trace()

    p_list = []
    for p in generator.parameters():
        p_list.append(p.grad)

    p_list_2 = []
    for p in discriminator.parameters():
        p_list_2.append(p.grad) 

    import pdb; pdb.set_trace()

    generator_optimizer.step()
    discriminator_optimizer.step()

    #if print_gradient:
        
    #    import pdb; pdb.set_trace
     #   g_grad = 0
     #   for p in generator.parameters():
     #       if p.grad:
     #           param_norm = p.grad.data.norm(2)
     #           g_grad += param_norm.item() ** 2
      #  g_grad = g_grad ** (1. / 2)

       # import pdb; pdb.set_trace

        #d_grad = 0
        #for p in discriminator.parameters():
        #    if p.grad:
        #        param_norm = p.grad.data.norm(2)
        #        d_grad += param_norm.item() ** 2
        #d_grad = d_grad ** (1. / 2)

      #  import pdb; pdb.set_trace

       # return g_loss.item(), d_loss.item(), g_grad, g_grad
    #else:
    return g_loss.item(), d_loss.item(), generator.parameters(), discriminator.parameters()

def train_iters_gan(model_path, loss_path, data_train, generator, discriminator,
               n_epochs, batch_size, learning_rate, code_to_token_mapping, print_gradient=False, print_every=5,
               plot_every=20, save_every=2, teacher_forcing_ratio=0.5,
               device='cpu'):
    '''
    Run train iteration.

    Keyword arguments:
    data_train (Custom PyTorch Dataset) -- the training data
    generator -- the generator network
    discriminator -- the discriminator network
    n_epochs (int) -- the number of training epochs
    batch_size (int) -- the batch size
    learning_rate (float) -- the learning rate
    print_every (int) -- defines print status intervall
    plot_every (int) -- defines plotting intervall
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training

    Outputs:
    generator -- the trained generator
    discriminator -- the trained discriminator
    '''
    start = time.time()

    plot_losses = []
    print_g_loss_total = 0  # Reset every print_every
    print_d_loss_total = 0
    plot_g_loss_total = 0  # Reset every plot_every
    plot_d_loss_total = 0

    generator_optimizer = optim.AdamW(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    embedder = Embedder(generator.input_size, generator.hidden_size)

    g_loss_dict = {}
    d_loss_dict = {}

    for epoch in range(1, n_epochs + 1):

        g_loss_list = []
        d_loss_list = []

        p_list = []

        for batch in DataLoader(data_train, batch_size=batch_size):

            # Tensor dimensions need to be checked
            input_tensor = batch[:, 0, :].to(device)
            input_tensor = torch.t(input_tensor) # only needed for LSTM
            target_tensor = batch[:, 1, :].to(device)
            target_tensor = torch.t(target_tensor)

            g_loss, d_loss, generator_parameters, discriminator_parameters = \
                        train_gan(input_tensor, target_tensor, generator,
                                  discriminator, generator_optimizer,
                                  discriminator_optimizer, criterion,
                                  teacher_forcing_ratio, code_to_token_mapping, 
                                  device, embedder)
            
            import pdb; pdb.set_trace()

            for p in generator_parameters:
                p_list.append(p.grad)

            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)

            print_g_loss_total += g_loss
            print_d_loss_total += d_loss

        g_loss_dict[epoch] = g_loss_list
        d_loss_dict[epoch] = d_loss_list

        if epoch % save_every == 0:
            root, ext = os.path.splitext(model_path)
            epoch_path = root + '_' + str(epoch) + ext
            torch.save({
                'trained_generator': generator.state_dict(),
                'trained_discriminator': discriminator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()
                }, epoch_path)

            with io.open(loss_path, mode='w') as loss_file:
                json.dump(g_loss_dict, loss_file)
            with io.open(loss_path, mode='w') as loss_file:
                json.dump(d_loss_dict, loss_file)

        # TODO: print statements have to be defined for both loss statements
        if epoch % print_every == 0:
            print_g_loss_avg = print_g_loss_total / print_every
            print_g_loss_total = 0
            print('Generator: {:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                              epoch,
                                                              int(epoch/n_epochs*100),
                                                              print_g_loss_avg))

        if epoch % print_every == 0:
            print_d_loss_avg = print_d_loss_total / print_every
            print_d_loss_total = 0
            print('Discriminator: {:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                              epoch,
                                                              int(epoch/n_epochs*100),
                                                              print_d_loss_avg))

        #if epoch % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

    #showPlot(plot_losses)

    return generator, discriminator, generator_optimizer, discriminator_optimizer

def train_iters_onehot(model_path, loss_path, training_size, converter,
               n_epochs, seq_length, batch_size, learning_rate, print_every=5, 
               plot_every=20, save_every=2, device='cpu'):
    '''
    Run train iteration.

    Keyword arguments:
    data_train (Custom PyTorch Dataset) -- the training data
    encoder -- the encoder network
    decoder -- the decoder network
    n_epochs (int) -- the number of training epochs
    batch_size (int) -- the batch size
    learning_rate (float) -- the learning rate
    with_attention (bool) -- defines if attention network is applied
    print_every (int) -- defines print status intervall
    plot_every (int) -- defines plotting intervall
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training

    Outputs:
    encoder -- the trained encoder
    decoder -- the trained decoder
    '''
    start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.AdamW(converter.parameters(), lr=learning_rate)

    #criterion = nn.NLLLoss()
    criterion = nn.MSELoss()

    loss_dict = {}

    input_size = converter.fc1.in_features

    for epoch in range(1, n_epochs + 1):

        loss_list = []

        batch_number = int(training_size / batch_size)

        for i in range(batch_number):

            input_tensor = torch.rand([batch_size, seq_length, input_size], requires_grad=True, device=device)


            #https://discuss.pytorch.org/t/set-max-value-to-1-others-to-0/44350/2

            max_values = input_tensor.argmax(2)
            target_tensor = nn.functional.one_hot(max_values, num_classes=input_size).float().to(device)

            pred_tensor = converter(input_tensor)
            #pred_argmax = pred_tensor.argmax(2)

            loss = criterion(pred_tensor, target_tensor)

            # Backprop and Optimization
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            print_loss_total += loss
            plot_loss_total += loss

        #for batch in DataLoader(data_train, batch_size=batch_size):

        #    # Tensor dimensions need to be checked
        #    input_tensor = batch[:, 0, :].to(device)
        #    input_tensor = torch.t(input_tensor)
        #    target_tensor = batch[:, 1, :].to(device)
        #    target_tensor = torch.t(target_tensor)

        #    pred_tensor = converter(input_tensor)

        #    loss = criterion(pred_tensor, target_tensor)

        #    # Backprop and Optimization
        #    loss.backward()
        #    optimizer.step()

        #    loss_list.append(loss)

        #    print_loss_total += loss
        #    plot_loss_total += loss

        loss_dict[epoch] = loss_list

        if epoch % save_every == 0:

            import pdb; pdb.set_trace()

            root, ext = os.path.splitext(model_path)
            epoch_path = root + '_' + str(epoch) + ext
            torch.save({
                'trained_converter': converter.state_dict(),
                'converter_optimizer': optimizer.state_dict(),
                }, epoch_path)
            with io.open(loss_path, mode='w') as loss_file:
                json.dump(loss_dict, loss_file)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('{:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                    epoch,
                                                    int(epoch/n_epochs*100),
                                                    print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    return converter, optimizer


def train_seq2seq(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, with_attention,
          teacher_forcing_ratio, device):
    '''
    Train one input sequence, ie. one forward pass.

    Keyword arguments:
    input_tensor (torch.Tensor) -- the input data
    target_tensor (torch.Tensor) -- the target data
    encoder -- the encoder network
    decoder -- the decoder network
    encoder_optimizer -- the encoder optimization algorithm
    decoder_optimizer -- the decoder optimization algorithm
    criterion -- the loss function
    with_attention (bool) -- defines if attention network is applied
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training (cpu/gpu)

    Outputs:
    the loss averaged by target length (float)
    '''

    ###################
    #                 #
    #  Encoding Step  #
    #                 #
    ###################

    encoder_hidden = encoder.init_hidden_state()
    encoder_cell = encoder.init_cell_state()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # This needs to be checked; dimensions may be different
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    # Depends on structure of input_tensor
    batch_size = input_tensor.shape[1]

    # Batch size implementation needs to be checked
    encoder_outputs = torch.zeros(batch_size,
                                  target_length,
                                  encoder.hidden_size,
                                  device=device)

    loss = 0

    for ei in range(input_length):

        #import pdb; pdb.set_trace()

        encoder_output, encoder_hidden, encoder_cell = \
            encoder(input_tensor[ei], encoder_hidden, encoder_cell)

        #encoder_outputs[ei] = encoder_output[0, 0]
        # Adjustment batch_size > 1; needs to be checked
        for bi in range(batch_size):
            encoder_outputs[bi, ei] = encoder_output[bi, 0]
    
    ###################
    #                 #
    #  Decoding Step  #
    #                 #
    ###################

    # Create input tensor with SOS encoding
    #decoder_input = torch.tensor([[input_tensor[0]]], device=device)

    decoder_input = input_tensor[0].clone().detach()

    decoder_hidden = encoder_hidden

    # Is this correct? (Alternative: initialize cell state to 0)
    decoder_cell = encoder_cell

    use_teacher_forcing = True if \
        random.random() < teacher_forcing_ratio else False

    if with_attention:

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell, \
                    decoder_attention = decoder(decoder_input,
                                                decoder_hidden,
                                                decoder_cell,
                                                encoder_outputs)


                loss += criterion(decoder_output, target_tensor[di])

                # Teacher forcing
                decoder_input = target_tensor[di]

        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell, \
                    decoder_attention = decoder(decoder_input,
                                                decoder_hidden,
                                                decoder_cell,
                                                encoder_outputs)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                #import pdb; pdb.set_trace()

                loss += criterion(decoder_output, target_tensor[di])
    else:
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, \
                    decoder_cell = decoder(decoder_input,
                                           decoder_hidden,
                                           decoder_cell)

                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:

            for di in range(target_length):

                decoder_output, decoder_hidden, decoder_cell = \
                    decoder(decoder_input, decoder_hidden, decoder_cell)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di])

                # Change hard coding (EOS token)
                #if decoder_input.item() == 2:
                #    break

    ###############################
    #                             #
    #  Backprop and optimization  #
    #                             #
    ###############################

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def train_iters_seq2seq(model_path, loss_path, data_train, encoder, decoder,
               n_epochs, batch_size, learning_rate, with_attention=False,
               print_every=5, plot_every=20, save_every=2,
               teacher_forcing_ratio=0.5, device='cpu'):
    '''
    Run train iteration.

    Keyword arguments:
    data_train (Custom PyTorch Dataset) -- the training data
    encoder -- the encoder network
    decoder -- the decoder network
    n_epochs (int) -- the number of training epochs
    batch_size (int) -- the batch size
    learning_rate (float) -- the learning rate
    with_attention (bool) -- defines if attention network is applied
    print_every (int) -- defines print status intervall
    plot_every (int) -- defines plotting intervall
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training

    Outputs:
    encoder -- the trained encoder
    decoder -- the trained decoder
    '''
    start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    loss_dict = {}

    for epoch in range(1, n_epochs + 1):

        loss_list = []

        for batch in DataLoader(data_train, batch_size=batch_size):

            # Tensor dimensions need to be checked
            input_tensor = batch[:, 0, :].to(device)
            input_tensor = torch.t(input_tensor)
            target_tensor = batch[:, 1, :].to(device)
            target_tensor = torch.t(target_tensor)

            loss = train_seq2seq(input_tensor,
                        target_tensor,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        criterion,
                        with_attention,
                        teacher_forcing_ratio,
                        device)

            loss_list.append(loss)

            print_loss_total += loss
            plot_loss_total += loss

        loss_dict[epoch] = loss_list

        if epoch % save_every == 0:
            root, ext = os.path.splitext(model_path)
            epoch_path = root + '_' + str(epoch) + ext
            torch.save({
                'trained_encoder': encoder.state_dict(),
                'trained_decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict()
                }, epoch_path)
            with io.open(loss_path, mode='w') as loss_file:
                json.dump(loss_dict, loss_file)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('{:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                    epoch,
                                                    int(epoch/n_epochs*100),
                                                    print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    return encoder, decoder, encoder_optimizer, decoder_optimizer


###############################################################################

def train_iters_gan_test(model_path, loss_path, data_train, generator, discriminator,
               n_epochs, batch_size, learning_rate, print_every=5,
               plot_every=20, save_every=2, teacher_forcing_ratio=0.5,
               device='cpu'):
    '''
    Run train iteration.

    Keyword arguments:
    data_train (Custom PyTorch Dataset) -- the training data
    generator -- the generator network
    discriminator -- the discriminator network
    n_epochs (int) -- the number of training epochs
    batch_size (int) -- the batch size
    learning_rate (float) -- the learning rate
    print_every (int) -- defines print status intervall
    plot_every (int) -- defines plotting intervall
    teacher_forcing_ratio (float) -- the ratio according to which
                                     teacher forcing is used
    device (str) -- the device used for training

    Outputs:
    generator -- the trained generator
    discriminator -- the trained discriminator
    '''
    start = time.time()

    plot_losses = []
    print_g_loss_total = 0  # Reset every print_every
    print_d_loss_total = 0
    plot_g_loss_total = 0  # Reset every plot_every
    plot_d_loss_total = 0

    generator_optimizer = optim.AdamW(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    g_loss_dict = {}
    d_loss_dict = {}

    for epoch in range(1, n_epochs + 1):

        g_loss_list = []
        d_loss_list = []

        for batch in DataLoader(data_train, batch_size=batch_size):

            # Tensor dimensions need to be checked
            input_tensor = batch[:, 0, :].to(device)
            input_tensor = torch.t(input_tensor) # only needed for LSTM
            target_tensor = batch[:, 1, :].to(device)
            target_tensor = torch.t(target_tensor)

            #import pdb; pdb.set_trace()

            g_loss, d_loss = train_gan(input_tensor, target_tensor, generator,
                        discriminator, generator_optimizer,
                        discriminator_optimizer, criterion,
                        teacher_forcing_ratio, device)

            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)

            print_g_loss_total += g_loss
            print_d_loss_total += d_loss

        g_loss_dict[epoch] = g_loss_list
        d_loss_dict[epoch] = d_loss_list

        if epoch % save_every == 0:
            root, ext = os.path.splitext(model_path)
            epoch_path = root + '_' + str(epoch) + ext
            torch.save({
                'trained_generator': generator.state_dict(),
                'trained_discriminator': discriminator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()
                }, epoch_path)

            with io.open(loss_path, mode='w') as loss_file:
                json.dump(g_loss_dict, loss_file)
            with io.open(loss_path, mode='w') as loss_file:
                json.dump(d_loss_dict, loss_file)

        # TODO: print statements have to be defined for both loss statements
        if epoch % print_every == 0:
            print_g_loss_avg = print_g_loss_total / print_every
            print_g_loss_total = 0
            print('Generator: {:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                              epoch,
                                                              int(epoch/n_epochs*100),
                                                              print_g_loss_avg))

        if epoch % print_every == 0:
            print_d_loss_avg = print_d_loss_total / print_every
            print_d_loss_total = 0
            print('Discriminator: {:s} ({:d} {:d}%) {:.6f}'.format(timeSince(start,
                                                              epoch/n_epochs),
                                                              epoch,
                                                              int(epoch/n_epochs*100),
                                                              print_d_loss_avg))

        #if epoch % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

    #showPlot(plot_losses)

    return generator, discriminator, generator_optimizer, discriminator_optimizer