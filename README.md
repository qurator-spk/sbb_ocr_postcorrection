# OCR Post-Correction  

This is the repo accompanying the paper:

Schaefer & Neudecker (2020). A Two-Step Approach for Automatic OCR Post-Correction. In the Proceedings of LaTeCH-CLfL 2020.

***

### Important Notice

We are currently working on refactoring and testing the code. The code will be updated on a regular basis.

***

## Usage

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
