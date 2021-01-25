from collections import defaultdict
from copy import deepcopy
import io
import json
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import os
import re
import string
import sys
import time
import unicodedata

from .preprocessing.database import load_alignments_from_sqlite, save_alignments_to_sqlite

def sec_to_min(sec):
    '''
    Taken from PyTorch tutorial
    '''
    m = math.floor(sec / 60)
    sec -= m * 60
    return '{}min {}sec'.format(int(m), int(sec))


def timeSince(since, percent):
    '''
    Taken from PyTorch tutorial
    '''
    now = time.time()
    elapsed_sec = now - since
    remaining_sec = (elapsed_sec / (percent)) - elapsed_sec
    return '{:s} (- {:s})'.format(sec_to_min(elapsed_sec), sec_to_min(remaining_sec))


def showPlot(points):
    '''
    Taken from PyTorch tutorial
    '''

    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def find_max_mod(s, b, current_max_valid=0):
    '''
    Inspired by: https://stackoverflow.com/questions/47571407/finding-the-the-largest-number-in-a-list-that-its-modulo-with-a-given-number-is
    '''
    if isinstance(s, int):
        s = list(range(s-b, s+1))
        assert len(s) <= sys.getrecursionlimit(), 'Decrease batch size to avoid RecursionError'
    if s[0] % b == 0:
        if current_max_valid < s[0]:
            current_max_valid = s[0]
    if len(s) > 1:
        return find_max_mod(s[1:], b, current_max_valid)
    return current_max_valid

def create_incremental_context_alignment(aligned_corpus, context_size=4):
    '''

    '''

    aligned_corpus_deepcopy = deepcopy(aligned_corpus)

    for doc_name, aligned_doc in aligned_corpus.items():
        for page_id, aligned_page in aligned_doc.items():

            #import pdb; pdb.set_trace()
            #print(page_id)
            splitted_ocr_page, splitted_gt_page = split_into_adjacent_parts(page_id, aligned_page)
            aligned_context_ocr_page, aligned_context_gt_page = align_context(splitted_ocr_page=splitted_ocr_page, splitted_gt_page=splitted_gt_page, context_size=context_size)

            aligned_corpus_deepcopy[doc_name][page_id][0] = aligned_context_ocr_page
            aligned_corpus_deepcopy[doc_name][page_id][1] = aligned_context_gt_page

            #if len(splitted_ocr_page) > 0:
            #    break

    return aligned_corpus_deepcopy, splitted_ocr_page, splitted_gt_page, aligned_context_ocr_page, aligned_context_gt_page


def align_context(splitted_ocr_page, splitted_gt_page, context_size=4):
    '''
    '''
    aligned_context_ocr_page = []
    aligned_context_gt_page = []

    line_id = 1

    for adjacent_ocr, adjacent_gt in zip(splitted_ocr_page, splitted_gt_page):
        ocr_tokenized = re.split('<wb>|<lb>', adjacent_ocr)
        gt_tokenized = re.split('<wb>|<lb>', adjacent_gt)

        i = 0
        min_length = max(len(ocr_tokenized), len(gt_tokenized))

        if context_size > min_length:
            aligned_context_ocr_page.append([str(line_id), ' '.join(ocr_tokenized)])
            aligned_context_gt_page.append([str(line_id), ' '.join(gt_tokenized)])
            line_id += 1
        else:
            while (i + context_size) <= min_length:
                aligned_context_ocr_page.append([str(line_id), ' '.join(ocr_tokenized[i:i+context_size])])
                aligned_context_gt_page.append([str(line_id), ' '.join(gt_tokenized[i:i+context_size])])
                line_id += 1
                i += 1

    return aligned_context_ocr_page, aligned_context_gt_page


def split_into_adjacent_parts(page_id, aligned_page):
    '''
    '''
    splitted_ocr_page = []
    splitted_gt_page = []

    adjacent_ocr_lines = []
    adjacent_gt_lines = []

#    import pdb; pdb.set_trace()

    def insert_break_tags(line):
        '''
        '''
        tagged_line = re.sub(r'\s+(\/)', r'@\1', line)
        reversed_tagging = tagged_line.translate(str.maketrans({' ': '<wb>', '@': ' '}))
        return reversed_tagging + '<lb>'

    #if page_id == 'P0003':
    #    import pdb; pdb.set_trace()

    list_index = 0
    for ocr_line, gt_line in zip(aligned_page[0], aligned_page[1]):
        if len(adjacent_ocr_lines) != 0:
            if int(ocr_line[0]) -1 == last_id:
                adjacent_ocr_lines.append(ocr_line[1])
                adjacent_gt_lines.append(gt_line[1])
                last_id = int(ocr_line[0])
                list_index += 1
                if (list_index + 1) == len(aligned_page[0]):
                    splitted_ocr_page.append(insert_break_tags(' '.join(adjacent_ocr_lines)))
                    splitted_gt_page.append(insert_break_tags(' '.join(adjacent_gt_lines)))
                    break
            else:
                splitted_ocr_page.append(insert_break_tags(' '.join(adjacent_ocr_lines)))
                splitted_gt_page.append(insert_break_tags(' '.join(adjacent_gt_lines)))
                adjacent_ocr_lines = []
                adjacent_gt_lines = []
                adjacent_ocr_lines.append(ocr_line[1])
                adjacent_gt_lines.append(gt_line[1])
                last_id = int(ocr_line[0])
        else:
            adjacent_ocr_lines.append(ocr_line[1])
            adjacent_gt_lines.append(gt_line[1])
            last_id = int(ocr_line[0])
            list_index += 1



#    def split_line_ids_into_adjacent_parts(line_ids):
#        splitted_ids = []
#
#        row = []
#        for id in line_ids:
#            if len(row) != 0:
#                if id -1 == last_id:
#                    row.append(id)
#                    last_id = id
#                else:
#                    splitted_ids.append(row)
#                    row = []
#                    row.append(id)
#                    last_id = id
#            else:
#                row.append(id)
#                last_id = id
#        return splitted_ids

    return splitted_ocr_page, splitted_gt_page


def unsqueeze_corpus(input_dir, output_dir, save=False):
    '''
    May not be needed if dataset class is used.
    '''

    with io.open(input_dir, mode='r') as f_in:
        aligned_corpus = json.load(f_in)

    aligned_data = gather_aligned_sequences(aligned_corpus, only_similar=True)

    if save:
        save_alignments_to_sqlite(aligned_data, path=output_dir, append=False)

    return aligned_data

def add_seq_id_to_aligned_seq(aligned_seq):
    '''
    '''
    id_ = 0

    for ocr, gt in aligned_seq:
        yield ((str(id_), ocr), (str(id_), gt))
        id_ += 1


def get_file_paths(dir_name):
    '''
    Extract all files recursively.

    Keyword arguments:
    dir_name (str) -- the root directory

    Outputs:
    file_paths_total (list) -- the extracted file paths
    None -- if dir_name does not exist
    '''

    try:
        file_paths = sorted(os.listdir(dir_name))
        file_paths_total = []

        for entry in file_paths:
            full_path = os.path.join(dir_name, entry)

            if os.path.isdir(full_path):
                file_paths_total += get_file_paths(full_path)
            else:
                file_paths_total.append(full_path)
        return file_paths_total
    except FileNotFoundError as fe:
        print(fe)
        return None


def get_gt_path_subset(ocr_root_path, gt_root_path, ocr_type='calamari'):
    '''
    Extracts GT counterparts to created OCR files.

    Keyword arguments:
    ocr_root_path (str) -- contains the OCR directories
    gt_root_path (str) -- contains the GT TEI Files

    Outputs:
    gt_paths (list) -- the absolute GT paths
    '''
    ocr_paths = sorted(os.listdir(ocr_root_path))
    gt_paths = []
    for ocr_path in ocr_paths:
        ocr_path = os.path.join(ocr_root_path, ocr_path, ocr_type + '_ocr')
        for ocr_doc in sorted(os.listdir(ocr_path)):
            gt_path = os.path.join(gt_root_path, ocr_doc + '.TEI-P5.xml')
            if os.path.isfile(gt_path):
                gt_paths.append(gt_path)
            else:
                print('{} is not a file!'.format(gt_path))
    return gt_paths


def create_char_set(extra_chars=None):
    '''
    Create character set with optional extra characters.

    Keyword arguments:
    extra_chars (str) -- extra characters in string format (default: None)

    Outputs:
    The character set (str) with optional extra characters
    '''

    if not extra_chars:
        return string.printable
    else:
        return string.printable + extra_chars


def normalize_data_encoding(corpus, form='NFC'):
    '''
    Normalize string encoding.

    Keyword arguments:
    corpus (dict) -- the whole corpus
    form (str) -- the encoding normalization type (default: NFS)

    Outputs:
    normalized_corpus (dict) -- the normalized corpus
    '''

    normalized_corpus = {}
    for doc_id, doc in corpus.items():
        normalized_corpus[doc_id] = {}
        for page_id, page in doc.items():
            normalized_corpus[doc_id][page_id] = []
            for seq in page:
                if seq is not None:
                    normalized_corpus[doc_id][page_id].append(unicodedata.normalize(form, seq))
                else:
                    normalized_corpus[doc_id][page_id].append(None)
    return normalized_corpus


def save_alignments_to_txt_file(aligned_seq, path):
    '''
    Save alignments to txt file.

    For large amounts of data this function is inefficient. Use sqlite
    implementation instead.

    Keyword arguments:
    aligned_seq (list) -- the alignments
    path (str) -- the path of the txt file
    '''
    with io.open(path, mode='w') as f_out:
        for seq_pair in aligned_seq:
            f_out.write(seq_pair[0] + '\t' + seq_pair[1])


def gather_aligned_sequences(aligned_corpus, only_similar=True):
    '''
    Gather alignments, CER, Levenshtein and max distances, and similarity scores.

    Keyword arguments:
    aligned_corpus (dict) -- the alignment corpus
    only_similar (bool) -- defines if only similar sequence should be gathered
                           (default: True)

    Outputs
    aligned_sequences (list) -- the similar sequences
    '''
    aligned_sequences = []
    for doc_name, aligned_doc in aligned_corpus.items():
        #print(doc_name)
        for page_id, aligned_page in aligned_doc.items():
            #print(page_id)
            for ocr, gt, cer, levenshtein, min_distance, max_distance, similarity_value in zip(aligned_page[0], aligned_page[1], aligned_page[2], aligned_page[3], aligned_page[4], aligned_page[5], aligned_page[6]):
                ocr_id = ocr[0]
                ocr_seq = ocr[1]
                gt_id = gt[0]
                gt_seq = gt[1]
                #print(ocr_id)
                #print(gt_id)
                assert ocr_id == gt_id, 'OCR and GT sequence ID is not identical.'
                if only_similar is True:
                    if similarity_value == 1:
                        aligned_sequences.append((doc_name, page_id, ocr_id, ocr_seq, gt_seq, cer, int(levenshtein), min_distance, max_distance, similarity_value))
                else:
                    aligned_sequences.append((doc_name, page_id, ocr_id, ocr_seq, gt_seq, cer, int(levenshtein), min_distance, max_distance, similarity_value))

    return aligned_sequences

def reconstruct_aligned_corpus(aligned_sequences):
    '''
    '''
    aligned_corpus = defaultdict(defaultdict)

    for alignment in aligned_sequences:
        pass



def combine_sequences_to_str(aligned_sequences):
    '''
    '''
    combined_ocr_seq = ''
    combined_gt_seq = ''

    for aligned_ocr_sequence, aligned_gt_sequence in zip(aligned_sequences[0], aligned_sequences[1]):
        if len(combined_ocr_seq) == 0 and len(combined_gt_seq) == 0:
            combined_ocr_seq += aligned_ocr_sequence
            combined_gt_seq += aligned_gt_sequence
        else:
            combined_ocr_seq += ' ' + aligned_ocr_sequence
            combined_gt_seq += ' ' + aligned_gt_sequence

    return combined_ocr_seq, combined_gt_seq


def normalize_char_alignments(aligned_characters):
    '''
    '''
    ocr_char = ''
    gt_char = ''

    for o, g in aligned_characters:
        if o == None:
            ocr_char += '@'
        elif o != None:
            ocr_char += o

        if g == None:
            gt_char += '@'
        elif g != None:
            gt_char += g

    return ocr_char, gt_char
