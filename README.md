# OCR Post-Correction  

This repo contains code for OCR post-correction. It is developed as part of the Qurator project located at the State Library Berlin. We define OCR post-correction as a Machine Translation problem and approach via a two-step pipeline:

1. a Detector
2. a Translator

The Detector is responsible for detecting incorrect OCR sequences. These are forwarded to the Translator, which corrects the incorrect characters in them.

The approach is described in more detail in the following paper. Please consider citing it when using the code:

Schaefer & Neudecker (2020). A Two-Step Approach for Automatic OCR Post-Correction. In the Proceedings of LaTeCH-CLfL 2020.

~~~
@inproceedings{schaefer-neudecker-2020-ocr-postcorrection,
    title = "A Two-Step Approach for Automatic {OCR} Post-Correction",
    author = "Schaefer, Robin  and
      Neudecker, Clemens",
    booktitle = "Proceedings of the The 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.latechclfl-1.6",
    pages = "52--57"
}
~~~

***

### Important Notice

We are currently working on refactoring and testing the code. The code will be updated on a regular basis.

***

## Installation

For installation we recommend to conduct the following steps:

1. Create a virtual environment, e.g. using `virtualenv`.
2. Clone the repo using `git clone`.
3. Install the repo using pip: `pip install .`.

***

## Usage

Commands of this tool basically fall into three categories:

1. Data Preprocessing
2. Feature Extraction
3. OCR Post-Correction.

### Data Preprocessing

#### 1. Align Sequences

After parsing the XML, OCR and GT sequences have to be aligned.

~~~
align-sequences /path/to/OCR path/to/GT /path/to/aligned/data
~~~

#### 2. Apply Sliding Window (optional)

Following Amhrein & Clematide (2018) we apply a sliding window approach. This optional step
restructures the data such that each token is surrounded by one preceding and two subsequent tokens.

~~~
apply-sliding-window /path/to/aligned/data /path/to/aligned/data/sliding/window
~~~

#### 3. Filter for Language (optional)

We noted that having cross-linguistic data decreased the post-correction results. Filtering the data for the target language may help.

~~~
filter-language /path/to/aligned/data /path/to/aligned/data/filtered --target-lang de
~~~

#### 4. Split into training, validation and test sets

### Feature Extraction

### OCR Post-Correction

#### Correcting unseen data

~~~
predict-translator /path/to/ocr/data /path/to/gt/data /path/to/trained/model /path/to/model/hyperparameters /path/to/code-to-token/mapping /path/to/output/directory
~~~