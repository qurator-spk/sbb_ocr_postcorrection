# mt

This directory contains code for the machine translation inspired approach to
OCR post-correction.

## cli_correct.py

This CLI file contains commands related to training and testing the models used
in this package.

## models

### error_detector.py

This module contains the PyTorch detector models. So far we made use of LSTM
and GRU models, although we achieved better results with LSTMs. We planned to
build a CNN-based detector as well. However, this is left for the future.

### gan.py

We are currently working on this component.

### predict.py

This module contains code for prediction.

### seq2seq.py

This module contains the PyTorch translator models. Both encoder and decoder
are either based on LSTM or GRU. So far we achieved better results with LSTMs,
which is why the GRU models are not really tested. In addition the basic LSTM
decoder, you can also find an attention-based LSTM decoder. We strongly
recommend to use the latter.

### train.py

This module contains code for training the detector and translator models.
