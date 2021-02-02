import click
from collections import defaultdict
from copy import deepcopy
import io
import json
from langid.langid import LanguageIdentifier, model
import numpy as np
import os
import pickle
import random
import xml.sax

from .sequence_similarity import check_sequence_similarity, print_alignment_stats
from .database import load_alignments_from_sqlite, save_alignments_to_sqlite
from .xml_parser import clean_tei, convert_to_page_id, \
    create_ocr_gt_id_mappings, extract_page_fulltext, TEIHandler

from qurator.sbb_ocr_postcorrection.data_structures import Corpus
from qurator.sbb_ocr_postcorrection.feature_extraction.encoding import add_padding
from qurator.sbb_ocr_postcorrection.helpers import add_seq_id_to_aligned_seq, \
    align_context, combine_sequences_to_str, \
    create_incremental_context_alignment, gather_aligned_sequences, \
    get_file_paths, get_gt_path_subset, normalize_char_alignments, \
    normalize_data_encoding, split_into_adjacent_parts, unsqueeze_corpus
from qurator.dinglehopper.align import align, seq_align


@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=False))
def align_sequences(ocr_dir, gt_dir, out_dir):
    '''
    Align OCR and GT sequences.

    \b
    Arguments:
    ocr-dir -- The absolute path to the OCR json file.
    gt-dir -- The absolute path to the GT json file.
    out-dir -- The absolute path to the aligned seq json file.
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    out_dir = os.path.abspath(out_dir)

    print_doc_stats = True
    print_page_stats = False

    char_alignment = False

    with io.open(ocr_dir, mode='r') as f_in:
        ocr_data = json.load(f_in)
    with io.open(gt_dir, mode='r') as f_in:
        gt_data = json.load(f_in)

    ocr_data = normalize_data_encoding(ocr_data, form='NFC')
    gt_data = normalize_data_encoding(gt_data, form='NFC')

    ###################################
    #                                 #
    #  GT and OCR Sequence Alignment  #
    #                                 #
    ###################################

    total_similar_sequences = 0
    total_sequences = 0
    aligned_corpus = defaultdict(defaultdict)
    if char_alignment:
        char_aligned_corpus = defaultdict(defaultdict)

    print('\nSTART: Sequence Alignment')
    for ocr_doc_id, gt_doc_id in zip(ocr_data, gt_data):
        assert ocr_doc_id == gt_doc_id, 'OCR Doc ID and GT Doc ID are not identical: {} (OCR) | {} (GT).'.format(ocr_doc_id, gt_doc_id)
        if print_page_stats:
            print('\n\nDocument ID: {}'.format(ocr_doc_id))
        doc_similar_sequences = 0
        doc_sequences = 0
        aligned_doc = defaultdict(list)

        for ocr_page_id, gt_page_id in zip(ocr_data[ocr_doc_id], gt_data[gt_doc_id]):
            # pre-check if IDs are okay and pages contain text
            assert ocr_page_id == gt_page_id, 'OCR Page ID and GT Page ID are not identical: {} (OCR) / {} (GT).'.format(ocr_page_id, gt_page_id)
            gt_page_length = len(gt_data[gt_doc_id][gt_page_id])
            ocr_page_length = len(ocr_data[ocr_doc_id][ocr_page_id])
            if gt_page_length == 0 or ocr_page_length == 0:
                continue

            # sequence alignment and similarity check
            aligned_sequences = seq_align(ocr_data[ocr_doc_id][ocr_page_id], gt_data[gt_doc_id][gt_page_id])
            aligned_sequences_with_id = add_seq_id_to_aligned_seq(aligned_sequences)
            ocr, gt, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding = check_sequence_similarity(aligned_sequences_with_id, similarity_range=(0.00, 0.10))
            assert len(ocr) == len(gt) == len(similarity_encoding), '# of OCR and GT sequences are not identical: {} (OCR) | {} (GT).'.format(ocr, gt)

            # some stats
            doc_sequences += gt_page_length
            total_sequences += gt_page_length
            num_similar_sequences = sum(similarity_encoding)
            doc_similar_sequences += num_similar_sequences
            total_similar_sequences += num_similar_sequences
            if print_page_stats:
                print_alignment_stats(ocr_page_id, gt_page_length, num_similar_sequences, scope='PAGE')

            # optional: char alignment
            if char_alignment:
                ocr_char_aligned = []
                gt_char_aligned = []
                for ocr_seq, gt_seq in zip(ocr, gt):
                    aligned_characters = align(ocr_seq, gt_seq)
                    ocr_char_aligned_seq, gt_char_aligned_seq = normalize_char_alignments(aligned_characters)

                    ocr_char_aligned.append(ocr_char_aligned_seq)
                    gt_char_aligned.append(gt_char_aligned_seq)

                    aligned_doc[ocr_page_id] = [ocr_char_aligned, gt_char_aligned, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding]
            else:
                aligned_doc[ocr_page_id] = (ocr, gt, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding)

    #        break
            #combined_ocr_seq, combined_gt_seq = combine_sequences_to_str(aligned_doc[ocr_page_id])

        if print_doc_stats:
            print_alignment_stats(ocr_doc_id, doc_sequences, doc_similar_sequences, scope='DOC')
        aligned_corpus[ocr_doc_id] = aligned_doc

    print('\nEND: Sequence Alignment')
    print_alignment_stats('DTA', total_sequences, total_similar_sequences, scope='CORPUS')

    with io.open(out_dir, mode='w') as f_out:
        json.dump(aligned_corpus, f_out)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=False))
def apply_sliding_window(in_dir, out_dir):
    '''
    Apply sliding window reformatting to aligned data.

    \b
    Arguments:
    in-dir -- The absolute path to the aligned JSON data
    out-dir -- The absolute path to the aligned JSON data (sliding window)
    '''

    # START: script

    # make paths absolute
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    with io.open(in_dir, mode='r') as f_in:
        aligned_corpus = json.load(f_in)

    aligned_corpus_context_aligned, splitted_ocr_page, splitted_gt_page, aligned_context_ocr_page, aligned_context_gt_page = create_incremental_context_alignment(aligned_corpus)

    # Helper functions should be moved elsewhere (in the long run)
    def generator(page):
        for ocr_line, gt_line in zip(page[0], page[1]):
            yield ((ocr_line[0], ocr_line[1]), (gt_line[0], gt_line[1]))

    aligned_corpus_context_aligned_copy = deepcopy(aligned_corpus_context_aligned)
    aligned_corpus_new = defaultdict(defaultdict)

    faulty_pages_total = {}

    for doc_id, doc_content in aligned_corpus_context_aligned.items():

        faulty_pages_doc = []

        print('Document ID: {}'.format(doc_id))
        aligned_doc = defaultdict(list)
        for page_id, page_content in doc_content.items():
            #print(page_id)
            page_iterator = generator(page_content)
            try:
                ocr, gt, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding = check_sequence_similarity(page_iterator, similarity_range=(0.00, 0.10))
                aligned_doc[page_id] = (ocr, gt, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding)
            except:
                faulty_pages_doc.append(page_id)
        aligned_corpus_new[doc_id] = aligned_doc

        faulty_pages_total[doc_id] = faulty_pages_doc
        #break

    with io.open(out_dir, mode='w') as f_out:
        json.dump(aligned_corpus_new, f_out)

################################################################################
@click.command()
@click.argument('training-dir', type=click.Path(exists=True))
@click.argument('validation-dir', type=click.Path(exists=True))
@click.argument('testing-dir', type=click.Path(exists=True))
@click.argument('training-target-dir', type=click.Path(exists=True))
@click.argument('validation-target-dir', type=click.Path(exists=True))
@click.argument('testing-target-dir', type=click.Path(exists=True))
def create_detector_targets(training_dir, validation_dir, testing_dir,
                            training_target_dir, validation_target_dir,
                            testing_target_dir):
    '''
    Needs to checked!!!
    '''

    # make paths absolute
    training_dir = os.path.abspath(training_dir)
    validation_dir = os.path.abspath(validation_dir)
    testing_dir = os.path.abspath(testing_dir)
    training_target_dir = os.path.abspath(training_target_dir)
    validation_target_dir = os.path.abspath(validation_target_dir)
    testing_target_dir = os.path.abspath(testing_target_dir)

    max_wordpiece_length = 1

    if max_wordpiece_length > 1:
        encoded_training_ocr_path = home_dir + '/Qurator/used_data/features/dta/encoded_training_ocr_sliding_window_3_charge2_170920.npy'
        encoded_training_gt_path = home_dir + '/Qurator/used_data/features/dta/encoded_training_gt_sliding_window_3_charge2_170920.npy'
        encoded_testing_ocr_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_ocr_sliding_window_3_2charges_170920.npy'
        encoded_testing_gt_path = home_dir + '/Qurator/used_data/features/dta/encoded_testing_gt_sliding_window_3_2charges_170920.npy'
        encoded_validation_ocr_path = home_dir + '/Qurator/used_data/features/dta/encoded_validation_ocr_sliding_window_3_2charges_small_170920.npy'
        encoded_validation_gt_path = home_dir + '/Qurator/used_data/features/dta/encoded_validation_gt_sliding_window_3_2charges_small_170920.npy'

        detector_training_path = home_dir + '/Qurator/used_data/features/dta/detector_target_training_sliding_window_german_3_charge2_170920.npy'
        detector_testing_path = home_dir + '/Qurator/used_data/features/dta/detector_target_testing_sliding_window_german_3_2charges_170920.npy'
        detector_validation_path = home_dir + '/Qurator/used_data/features/dta/detector_target_validation_sliding_window_german_3_2charges_small_170920.npy'

        encoded_training_ocr = np.load(encoded_training_ocr_path)
        encoded_training_gt = np.load(encoded_training_gt_path)
        encoded_testing_ocr = np.load(encoded_testing_ocr_path)
        encoded_testing_gt = np.load(encoded_testing_gt_path)
        encoded_validation_ocr = np.load(encoded_validation_ocr_path)
        encoded_validation_gt = np.load(encoded_validation_gt_path)
    else:
        alignments_training, _, _ = load_alignments_from_sqlite(path=training_dir, size='total')
        alignments_testing, _, _ = load_alignments_from_sqlite(path=testing_dir, size='total')
        alignments_validation, _, _ = load_alignments_from_sqlite(path=validation_dir, size='total')

    if max_wordpiece_length == 1:

        max_length = 100

        training_targets = []
        testing_targets = []
        validation_targets = []

        # targets training
        for alignment in alignments_training:
            ocr = alignment[3]
            gt = alignment[4]

            if len(ocr) != len(gt):
                diff = abs(len(ocr)-len(gt))
                if len(ocr) < len(gt):
                    ocr += (diff*' ')
                else:
                    gt += (diff*' ')

            assert len(ocr) == len(gt)

            training_target = []

            for char_ocr, char_gt in zip(ocr, gt):
                if char_ocr == char_gt:
                    training_target.append(1)
                else:
                    training_target.append(2)
            training_targets.append(training_target)

        # targets testing
        for alignment in alignments_testing:
            ocr = alignment[3]
            gt = alignment[4]


            if len(ocr) != len(gt):
                diff = abs(len(ocr)-len(gt))
                if len(ocr) < len(gt):
                    ocr += (diff*' ')
                else:
                    gt += (diff*' ')

            assert len(ocr) == len(gt)

            testing_target = []

            for char_ocr, char_gt in zip(ocr, gt):
                if char_ocr == char_gt:
                    testing_target.append(1)
                else:
                    testing_target.append(2)
            testing_targets.append(testing_target)

        # targets validation
        for alignment in alignments_validation:
            ocr = alignment[3]
            gt = alignment[4]


            if len(ocr) != len(gt):
                diff = abs(len(ocr)-len(gt))
                if len(ocr) < len(gt):
                    ocr += (diff*' ')
                else:
                    gt += (diff*' ')

            assert len(ocr) == len(gt)

            validation_target = []

            for char_ocr, char_gt in zip(ocr, gt):
                if char_ocr == char_gt:
                    validation_target.append(1)
                else:
                    validation_target.append(2)
            validation_targets.append(validation_target)

        training_targets = add_padding(training_targets, max_length)
        testing_targets = add_padding(testing_targets, max_length)
        validation_targets = add_padding(validation_targets, max_length)

        np.save(training_target_dir, training_targets)
        np.save(testing_target_dir, testing_targets)
        np.save(validation_target_dir, validation_targets)
    else:
        training_targets = []
        testing_targets = []
        validation_targets = []

        # create training targets
        for sequence_id in range(encoded_training_ocr.shape[0]):
            targets_sequence = []
            for encoding_ocr, encoding_gt in zip(encoded_training_ocr[sequence_id], encoded_training_gt[sequence_id]):
                if encoding_gt == 0:
                    targets_sequence.append(0)
                elif encoding_ocr == encoding_gt:
                    targets_sequence.append(1)
                else:
                    targets_sequence.append(2)
            training_targets.append(targets_sequence)

        training_targets = np.array(training_targets)

        np.save(training_target_dir, training_targets)

        # create testing targets
        for sequence_id in range(encoded_testing_ocr.shape[0]):
            targets_sequence = []
            for encoding_ocr, encoding_gt in zip(encoded_testing_ocr[sequence_id], encoded_testing_gt[sequence_id]):
                if encoding_gt == 0:
                    targets_sequence.append(0)
                elif encoding_ocr == encoding_gt:
                    targets_sequence.append(1)
                else:
                    targets_sequence.append(2)
            testing_targets.append(targets_sequence)

        testing_targets = np.array(testing_targets)

        np.save(testing_target_dir, testing_targets)

        # create validation targets
        for sequence_id in range(encoded_validation_ocr.shape[0]):
            targets_sequence = []
            for encoding_ocr, encoding_gt in zip(encoded_validation_ocr[sequence_id], encoded_validation_gt[sequence_id]):
                if encoding_gt == 0:
                    targets_sequence.append(0)
                elif encoding_ocr == encoding_gt:
                    targets_sequence.append(1)
                else:
                    targets_sequence.append(2)
            validation_targets.append(targets_sequence)

        validation_targets = np.array(validation_targets)

        np.save(validation_target_dir, validation_targets)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=False))
@click.option('--target-lang', default='de', help='The target language, i.e. the language to be kept.')
def filter_language(in_dir, out_dir, target_lang):
    '''
    Apply language filter to aligned data.

    \b
    Arguments:
    in-dir -- The absolute path to the aligned data (JSON)
    out-dir -- The absolute path to the filtered data (DB)
    '''

    # make paths absolute
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    with io.open(in_dir, mode='r') as f_in:
        aligned_corpus = json.load(f_in)

    corpus = Corpus()

    for doc_id, doc in aligned_corpus.items():
        corpus.add_doc(doc_id, doc)

    corpus.convert_to_sqlite_format()

#    unfiltered_data = unsqueeze_corpus(in_dir, out_dir, save=False)

    #loaded_data, loaded_data_as_df, headers = load_alignments_from_sqlite(path=input_path, size='total')
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    len_total_corpus = 0
    len_german_corpus = 0

#    for doc_name, aligned_doc in aligned_corpus.items():
#        for page_id, aligned_page in aligned_doc.items():
#            new_ocr_page = []
#            new_gt_page = []
#            new_cer = []
#            new_levenshtein = []
#            new_min_distance = []
#            new_max_distance = []
#            new_similarity_encoding = []
#            for ocr_line, gt_line, cer, levenshtein, min_distance, max_distance, similarity_encoding in zip(aligned_page[0],
#                                                                                                            aligned_page[1],
#                                                                                                            aligned_page[2],
#                                                                                                            aligned_page[3],
#                                                                                                            aligned_page[4],
#                                                                                                            aligned_page[5],
#                                                                                                            aligned_page[6]):
#                lang, prob = identifier.classify(gt_line[1])
#
#                if lang == target_lang and prob > 0.999:
#                    new_ocr_page.append(ocr_line)
#                    new_gt_page.append(gt_line)
#                    new_cer.append(cer)
#                    new_levenshtein.append(levenshtein)
#                    new_min_distance.append(min_distance)
#                    new_max_distance.append(max_distance)
#                    new_similarity_encoding.append(similarity_encoding)
#
#            filtered_data = [new_ocr_page, new_gt_page, new_cer, new_levenshtein,
#                             new_min_distance, new_max_distance, new_similarity_encoding]
#
#            for i in range(len(filtered_data)):
#                aligned_corpus[doc_name][page_id][i] = filtered_data[i]
#
#            len_total_corpus += len(aligned_page[0])
#            len_german_corpus += len(new_ocr_page)

    filtered_data = []
    for i, alignment in enumerate(corpus.aligned_sequences):
        gt_seq = alignment[4] # GT is taken as it is supposed to contain fewer errors than the OCR
        lang, prob = identifier.classify(gt_seq)

        if lang == target_lang and prob > 0.999:
            filtered_data.append(alignment)

        if i % 10000 == 0:
            print('Language-filtered files: {}'.format(i))

    print('Non-filtered data: {}'.format(len(unfiltered_data)))
    print('Filtered data: {}'.format(len(filtered_data)))

    save_alignments_to_sqlite(filtered_data, path=out_dir, append=False)

################################################################################
@click.command()
@click.argument('ocr-dir', type=click.Path(exists=True))
@click.argument('gt-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=False))
#@click.argument('out-ocr-dir', type=click.Path(exists=True))
#@click.argument('out-gt-dir', type=click.Path(exists=True))
def parse_xml(ocr_dir, gt_dir, out_dir):
    '''
    Parse OCR and GT XML and save respective JSON files to output directory.

    \b
    Arguments:
    ocr-dir -- OCR root path
    gt-dir -- GT root path
    out-dir -- The directory of the output json
    '''

    # make paths absolute
    ocr_dir = os.path.abspath(ocr_dir)
    gt_dir = os.path.abspath(gt_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_ocr_dir = os.path.join(out_dir, 'ocr_data.json')
    out_gt_dir = os.path.join(out_dir, 'gt_data.json')

    ##########################
    #                        #
    #  Parse OCR XML (PAGE)  #
    #                        #
    ##########################

    parsed_ocr_corpus = defaultdict(defaultdict)
    parsed_ocr_documents = 0
    mets_paths = []
    print('\nSTART: DTA OCR XML Parsing')
    for dir in sorted(os.listdir(ocr_dir)):
        try:
            ocr_calamari_dir = os.path.join(ocr_dir, dir, 'calamari_ocr')
            for doc in sorted(os.listdir(ocr_calamari_dir)):
                calamari_doc_dir = os.path.join(ocr_calamari_dir, doc)
                mets_dir = os.path.join(calamari_doc_dir, 'mets.xml')
                mets_paths.append(mets_dir)
                ocr_id_mapping, _ = create_ocr_gt_id_mappings(mets_dir)  # gt_id_mapping will be used for GT parsing
                parsed_ocr_corpus[doc], f_paths = extract_page_fulltext(calamari_doc_dir, ocr_id_mapping, conf_threshold=None)
                parsed_ocr_documents += 1
        except FileNotFoundError as fe:
            print(fe)

    with io.open(out_ocr_dir, mode='w') as f_out:
        json.dump(parsed_ocr_corpus, f_out)

    print('END: DTA OCR XML Parsing')
    print('Total # of parsed OCR documents: {}'.format(parsed_ocr_documents))

    ########################
    #                      #
    #  Parse GT XML (TEI)  #
    #                      #
    ########################

    gt_paths = get_gt_path_subset(ocr_dir, gt_dir)

    if not gt_paths:
        gt_paths = get_file_paths(gt_dir)

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    handler = TEIHandler()
    parser.setContentHandler(handler)

    parsed_gt_corpus = defaultdict(defaultdict)
    print('\nSTART: DTA GT XML Parsing')
    for i, f_path in enumerate(gt_paths):
        document_name = f_path.split('/')[-1].split('.')[0]
        parser.parse(f_path)
        for page in handler.pages:
            handler.pages[page] = clean_tei(handler.pages[page])
        parsed_gt_corpus[document_name] = handler.pages
        handler.__init__()

    # quite ugly fix to convert img IDs to page IDs
    converted_gt_corpus = defaultdict()
    for mets_path in mets_paths:
        _, gt_id_mapping = create_ocr_gt_id_mappings(mets_path)
        doc = mets_path.split('/')[-2]
        converted_doc = convert_to_page_id(parsed_gt_corpus[doc], gt_id_mapping)
        converted_gt_corpus[doc] = converted_doc

        if i+1 % 250 == 0:
            print('%d books processed' % (i+1))
    print('END: DTA GT XML Parsing')
    print('Total # of parsed GT documents: {}'.format(i+1))

    # Save GT as JSON
    with io.open(out_gt_dir, mode='w') as f_out:
        json.dump(converted_gt_corpus, f_out)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.option('--training-proportion', default=0.8, help='The training proportion of the dataset.')
def split_dataset(in_dir, out_dir, training_proportion):
    '''
    Formerly 'run_dataset_splitting_charge1.py'

    in-dir -- Input database
    out-dir -- Path to output databases
    '''

    # make paths absolute
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    training_dir = os.path.join(out_dir, 'training_set.db')
    testing_dir = os.path.join(out_dir, 'testing_set.db')
    validation_dir = os.path.join(out_dir, 'validation_set.db')

    loaded_data, _, _ = load_alignments_from_sqlite(path=in_dir, size='total')

    training_size = int(len(loaded_data) * training_proportion)

    zero = 0

    for a in loaded_data:
        if a[5] == 0:
            zero+=1

    #CER statistics
    no_cer = 0
    tiny_cer = 0
    low_cer = 0
    medium_cer = 0
    high_cer = 0
    huge_cer = 0
    enormous_cer = 0

    no_cer_set = []
    tiny_cer_set = []
    low_cer_set = []
    medium_cer_set = []
    high_cer_set = []
    huge_cer_set = []
    enormous_cer_set = []

    for a in loaded_data:
        if a[5] == 0:
            no_cer+=1
            no_cer_set.append(a)
        elif a[5] < 0.02:
            tiny_cer+=1
            tiny_cer_set.append(a)
        elif a[5] < 0.04:
            low_cer+=1
            low_cer_set.append(a)
        elif a[5] < 0.06:
            medium_cer+=1
            medium_cer_set.append(a)
        elif a[5] < 0.08:
            high_cer+=1
            high_cer_set.append(a)
        elif a[5] < 0.1:
            huge_cer+=1
            huge_cer_set.append(a)
        else:
            enormous_cer+=1
            enormous_cer_set.append(a)


    print('\nCER STATISTICS')
    print('No CER: {}'.format(no_cer))
    print('Tiny CER (<0.02): {}'.format(tiny_cer))
    print('Low CER (<0.04): {}'.format(low_cer))
    print('Medium CER (<0.06): {}'.format(medium_cer))
    print('High CER (<0.08): {}'.format(high_cer))
    print('Huge CER (<0.1): {}'.format(huge_cer))
    print('Enormous CER (>=0.1): {}'.format(enormous_cer))

    #proportional splitting
    corpus_size = len(loaded_data)
    corpus_size = len(no_cer_set) + len(tiny_cer_set) + len(low_cer_set) + \
                  len(medium_cer_set) + len(high_cer_set) + len(huge_cer_set)

    no_prop = round(no_cer/corpus_size, 3)
    tiny_prop = round(tiny_cer/corpus_size, 3)
    low_prop = round(low_cer/corpus_size, 3)
    medium_prop = round(medium_cer/corpus_size, 3)
    high_prop = round(high_cer/corpus_size, 3)
    huge_prop = round(huge_cer/corpus_size, 3)
    #enormous_prop = round(enormous_cer/corpus_size, 3)

    assert 1.0 == (no_prop +
                   tiny_prop +
                   low_prop +
                   medium_prop +
                   high_prop +
                   huge_prop)# +
        #           enormous_prop)

    # training sizes (by proportions)
    no_training_size = int(no_prop * training_size)
    tiny_training_size = int(tiny_prop * training_size)
    low_training_size = int(low_prop * training_size)
    medium_training_size = int(medium_prop * training_size)
    high_training_size = int(high_prop * training_size)
    huge_training_size = int(huge_prop * training_size)
    #enormous_training_size = int(enormous_prop * training_size)

    assert training_size == (no_training_size +
                      tiny_training_size +
                      low_training_size +
                      medium_training_size +
                      high_training_size +
                      huge_training_size)# +
    #                  enormous_training_size)

    random.seed(49)

    random.shuffle(no_cer_set)
    random.shuffle(tiny_cer_set)
    random.shuffle(low_cer_set)
    random.shuffle(medium_cer_set)
    random.shuffle(high_cer_set)
    random.shuffle(huge_cer_set)
    #random.shuffle(enormous_cer_set)

    no_training = no_cer_set[:no_training_size]
    tiny_training = tiny_cer_set[:tiny_training_size]
    low_training = low_cer_set[:low_training_size]
    medium_training = medium_cer_set[:medium_training_size]
    high_training = high_cer_set[:high_training_size]
    huge_training = huge_cer_set[:huge_training_size]
    #enormous_training = enormous_cer_set[:enormous_training_size]

    training_set = no_training + tiny_training + low_training + medium_training + high_training + huge_training# + enormous_training

    assert training_size == (len(no_training) +
                             len(tiny_training) +
                             len(low_training) +
                             len(medium_training) +
                             len(high_training) +
                             len(huge_training))# +
                             #len(enormous_training))

    no_testing = no_cer_set[no_training_size:]
    tiny_testing = tiny_cer_set[tiny_training_size:]
    low_testing = low_cer_set[low_training_size:]
    medium_testing = medium_cer_set[medium_training_size:]
    high_testing = high_cer_set[high_training_size:]
    huge_testing = huge_cer_set[huge_training_size:]
#    enormous_testing = enormous_cer_set[enormous_training_size:]

    testing_set = no_testing + tiny_testing + low_testing + medium_testing + high_testing + huge_testing# + enormous_testing

    assert corpus_size - training_size == (len(no_testing) +
                                           len(tiny_testing) +
                                           len(low_testing) +
                                           len(medium_testing) +
                                           len(high_testing) +
                                           len(huge_testing))# +
                                           #len(enormous_testing))

    random.shuffle(training_set)
    random.shuffle(testing_set)

    validation_size = int(len(testing_set)/2)

    validation_set = testing_set[:validation_size]
    testing_set = testing_set[validation_size:]

    save_alignments_to_sqlite(training_set, path=training_dir, append=False)
    save_alignments_to_sqlite(testing_set, path=testing_dir, append=False)
    save_alignments_to_sqlite(validation_set, path=validation_dir, append=False)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
@click.option('--training-proportion', default=0.8, help='The training proportion of the dataset.')
def split_dataset_2(in_dir, out_dir, training_proportion):
    '''
    NOT TESTED! DO NOT USE! (formerly, run_dataset_splitting_charge2)

    Arguments.
    in-dir -- Input database
    out-dir -- Path to output databases
    '''

    input_path = home_dir + '/Qurator/used_data/preproc_data/dta/aligned_corpus_german_charge2_110920.db'
    training_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_german_2charges_110920.db'
    testing_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_german_2charges_110920.db'
    validation_path = home_dir + '/Qurator/used_data/preproc_data/dta/validation_set_german_2charges_110920.db'

    loaded_data, loaded_data_as_df, headers = load_alignments_from_sqlite(path=input_path, size='total')

    combine_with_charge1 = True

    if combine_with_charge1:
        training_charge1_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_german_charge1_110920.db'
        validation_charge1_path = home_dir + '/Qurator/used_data/preproc_data/dta/validation_set_german_charge1_110920.db'
        testing_charge1_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_german_charge1_110920.db'

        training_charge1_data, training_charge1_data_as_df, headers_charge1 = load_alignments_from_sqlite(path=training_charge1_path, size='total')
        testing_charge1_data, testing_charge1_data_as_df, headers_charge1 = load_alignments_from_sqlite(path=testing_charge1_path, size='total')
        validation_charge1_data, validation_charge1_data_as_df, headers_charge1 = load_alignments_from_sqlite(path=validation_charge1_path, size='total')

    training_size = 70000

    zero = 0

    for a in loaded_data:
        if a[5] == 0:
            zero+=1

    #CER statistics
    no_cer = 0
    tiny_cer = 0
    low_cer = 0
    medium_cer = 0
    high_cer = 0
    huge_cer = 0
    enormous_cer = 0

    no_cer_set = []
    tiny_cer_set = []
    low_cer_set = []
    medium_cer_set = []
    high_cer_set = []
    huge_cer_set = []
    enormous_cer_set = []

    for a in loaded_data:
        if a[5] == 0:
            no_cer+=1
            no_cer_set.append(a)
        elif a[5] < 0.02:
            tiny_cer+=1
            tiny_cer_set.append(a)
        elif a[5] < 0.04:
            low_cer+=1
            low_cer_set.append(a)
        elif a[5] < 0.06:
            medium_cer+=1
            medium_cer_set.append(a)
        elif a[5] < 0.08:
            high_cer+=1
            high_cer_set.append(a)
        elif a[5] < 0.1:
            huge_cer+=1
            huge_cer_set.append(a)
        else:
            enormous_cer+=1
            enormous_cer_set.append(a)

    print('\nCER STATISTICS (CHARGE 2)')
    print('No CER: {}'.format(no_cer))
    print('Tiny CER (<0.02): {}'.format(tiny_cer))
    print('Low CER (<0.04): {}'.format(low_cer))
    print('Medium CER (<0.06): {}'.format(medium_cer))
    print('High CER (<0.08): {}'.format(high_cer))
    print('Huge CER (<0.1): {}'.format(huge_cer))
    print('Enormous CER (>=0.1): {}'.format(enormous_cer))

    if combine_with_charge1:
        #CER statistics
        no_cer_charge1 = 0
        tiny_cer_charge1 = 0
        low_cer_charge1 = 0
        medium_cer_charge1 = 0
        high_cer_charge1 = 0
        huge_cer_charge1 = 0
        enormous_cer_charge1 = 0

        no_cer_set_charge1 = []
        tiny_cer_set_charge1 = []
        low_cer_set_charge1 = []
        medium_cer_set_charge1 = []
        high_cer_set_charge1 = []
        huge_cer_set_charge1 = []
        enormous_cer_set_charge1 = []

        for a in training_charge1_data:
            if a[5] == 0:
                no_cer_charge1+=1
                no_cer_set_charge1.append(a)
            elif a[5] < 0.02:
                tiny_cer_charge1+=1
                tiny_cer_set_charge1.append(a)
            elif a[5] < 0.04:
                low_cer_charge1+=1
                low_cer_set_charge1.append(a)
            elif a[5] < 0.06:
                medium_cer_charge1+=1
                medium_cer_set_charge1.append(a)
            elif a[5] < 0.08:
                high_cer_charge1+=1
                high_cer_set_charge1.append(a)
            elif a[5] < 0.1:
                huge_cer_charge1+=1
                huge_cer_set_charge1.append(a)
            else:
                enormous_cer_charge1+=1
                enormous_cer_set_charge1.append(a)

        #CER statistics validation charge1
        no_cer_charge1_validation = 0
        tiny_cer_charge1_validation = 0
        low_cer_charge1_validation = 0
        medium_cer_charge1_validation = 0
        high_cer_charge1_validation = 0
        huge_cer_charge1_validation = 0
        enormous_cer_charge1_validation = 0

        no_cer_set_charge1_validation = []
        tiny_cer_set_charge1_validation = []
        low_cer_set_charge1_validation = []
        medium_cer_set_charge1_validation = []
        high_cer_set_charge1_validation = []
        huge_cer_set_charge1_validation = []
        enormous_cer_set_charge1_validation = []

        for a in validation_charge1_data:
            if a[5] == 0:
                no_cer_charge1_validation+=1
                no_cer_set_charge1_validation.append(a)
            elif a[5] < 0.02:
                tiny_cer_charge1_validation+=1
                tiny_cer_set_charge1_validation.append(a)
            elif a[5] < 0.04:
                low_cer_charge1_validation+=1
                low_cer_set_charge1_validation.append(a)
            elif a[5] < 0.06:
                medium_cer_charge1_validation+=1
                medium_cer_set_charge1_validation.append(a)
            elif a[5] < 0.08:
                high_cer_charge1_validation+=1
                high_cer_set_charge1_validation.append(a)
            elif a[5] < 0.1:
                huge_cer_charge1_validation+=1
                huge_cer_set_charge1_validation.append(a)
            else:
                enormous_cer_charge1_validation+=1
                enormous_cer_set_charge1_validation.append(a)


    #proportional splitting
    #corpus_size = len(incorrect_set)
    corpus_size = len(no_cer_set) + len(tiny_cer_set) + len(low_cer_set) + \
                  len(medium_cer_set) + len(high_cer_set) + len(huge_cer_set)

    no_prop = round(no_cer/corpus_size, 3)
    tiny_prop = round(tiny_cer/corpus_size, 3)
    low_prop = round(low_cer/corpus_size, 3)
    medium_prop = round(medium_cer/corpus_size, 3)
    high_prop = round(high_cer/corpus_size, 3)
    huge_prop = round(huge_cer/corpus_size, 3)
    #enormous_prop = round(enormous_cer/corpus_size, 3)

    assert 1.0 == (no_prop +
                   tiny_prop +
                   low_prop +
                   medium_prop +
                   high_prop +
                   huge_prop)# +
        #           enormous_prop)

    # training sizes (by proportions)
    no_training_size = int(no_prop * training_size)
    tiny_training_size = int(tiny_prop * training_size)
    low_training_size = int(low_prop * training_size)
    medium_training_size = int(medium_prop * training_size)
    high_training_size = int(high_prop * training_size)
    huge_training_size = int(huge_prop * training_size)
    #enormous_training_size = int(enormous_prop * training_size)

    assert training_size == (no_training_size +
                      tiny_training_size +
                      low_training_size +
                      medium_training_size +
                      high_training_size +
                      huge_training_size)# +
    #                  enormous_training_size)

    random.seed(49)

    random.shuffle(no_cer_set)
    random.shuffle(tiny_cer_set)
    random.shuffle(low_cer_set)
    random.shuffle(medium_cer_set)
    random.shuffle(high_cer_set)
    random.shuffle(huge_cer_set)
    #random.shuffle(enormous_cer_set)

    no_training = no_cer_set[:no_training_size]
    tiny_training = tiny_cer_set[:tiny_training_size]
    low_training = low_cer_set[:low_training_size]
    medium_training = medium_cer_set[:medium_training_size]
    high_training = high_cer_set[:high_training_size]
    huge_training = huge_cer_set[:huge_training_size]
    #enormous_training = enormous_cer_set[:enormous_training_size]

    training_set = no_training + tiny_training + low_training + medium_training + high_training + huge_training# + enormous_training

    assert training_size == (len(no_training) +
                             len(tiny_training) +
                             len(low_training) +
                             len(medium_training) +
                             len(high_training) +
                             len(huge_training))# +
                             #len(enormous_training))

    no_testing = no_cer_set[no_training_size:]
    tiny_testing = tiny_cer_set[tiny_training_size:]
    low_testing = low_cer_set[low_training_size:]
    medium_testing = medium_cer_set[medium_training_size:]
    high_testing = high_cer_set[high_training_size:]
    huge_testing = huge_cer_set[huge_training_size:]
#    enormous_testing = enormous_cer_set[enormous_training_size:]

    testing_set = no_testing + tiny_testing + low_testing + medium_testing + high_testing + huge_testing# + enormous_testing

    assert corpus_size - training_size == (len(no_testing) +
                                           len(tiny_testing) +
                                           len(low_testing) +
                                           len(medium_testing) +
                                           len(high_testing) +
                                           len(huge_testing))# +
                                           #len(enormous_testing))

    #random.shuffle(training_set)
    random.shuffle(testing_set)

    validation_size = int(len(testing_set)/2)

    validation_set = testing_set[:validation_size]
    testing_set = testing_set[validation_size:]

    no_cer_validation = []
    incorrect_cer_validation = []
    for a in validation_set:
        if a[5] == 0:
            no_cer_validation.append(a)
        else:
            incorrect_cer_validation.append(a)

    no_cer_validation_size = 640
    no_validation_sample = random.sample(no_cer_validation, no_cer_validation_size)
    incorrect_cer_validation.extend(no_validation_sample)
    validation_set = incorrect_cer_validation.copy()

    incorrect_training = tiny_training + low_training + medium_training + high_training + huge_training

    no_cer_training_size = 1950

    no_training_sample = random.sample(no_training, no_cer_training_size)

    incorrect_training.extend(no_training_sample)
    training_set = incorrect_training.copy()

    if combine_with_charge1:
        #training
        incorrect_set_charge1 = tiny_cer_set_charge1 + low_cer_set_charge1 + medium_cer_set_charge1 + high_cer_set_charge1 + huge_cer_set_charge1
        no_cer_training_size_charge1 = 3400
        no_training_sample_charge1 = random.sample(no_cer_set_charge1, no_cer_training_size_charge1)
        incorrect_set_charge1.extend(no_training_sample_charge1)
        training_set.extend(incorrect_set_charge1)

        #validation
        incorrect_set_charge1_validation = tiny_cer_set_charge1_validation + low_cer_set_charge1_validation + medium_cer_set_charge1_validation + high_cer_set_charge1_validation + huge_cer_set_charge1_validation
        no_cer_validation_size_charge1 = 700
        no_validation_sample_charge1 = random.sample(no_cer_set_charge1_validation, no_cer_validation_size_charge1)
        incorrect_set_charge1_validation.extend(no_validation_sample_charge1)
        validation_set.extend(incorrect_set_charge1_validation)

        #testing
        testing_set.extend(testing_charge1_data)

    random.shuffle(training_set)

    print('\nSize training set: {}'.format(len(training_set)))
    print('Size testing set: {}'.format(len(testing_set)))
    print('Size validation set: {}'.format(len(validation_set)))

    save_alignments_to_sqlite(training_set, path=training_path, append=False)
    save_alignments_to_sqlite(testing_set, path=testing_path, append=False)
    save_alignments_to_sqlite(validation_set, path=validation_path, append=False)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
#@click.option('training-proportion', default=0.8, help='Training data proportion')
@click.option('--seed', default=49, help='The seed of the random number generator.')
def split_dataset_sliding_window(in_dir, out_dir, seed):
    '''
    Split aligned data (sliding window) into training, validation and testing sets.

    \b
    Arguments:
    in-dir -- Input database
    out-dir -- Path to output databases

    Formerly run_dataset_splitting_sliding_window_charge1.py
    '''

    random.seed(seed)

    # make paths absolute
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    training_dir = os.path.join(out_dir, 'training_set_sliding_window.db')
    testing_dir = os.path.join(out_dir, 'testing_set_sliding_window.db')
    validation_dir = os.path.join(out_dir, 'validation_set_sliding_window.db')

    loaded_data, _, _ = load_alignments_from_sqlite(path=in_dir, size='total')

    # remove three word lines
    data_compact = []

    for line in loaded_data:
        if len(line[4].split(' ')) == 4:
            data_compact.append(line)

    # remove long lines

    data_short_lines = []

    for line in data_compact:
        if len(line[4]) <= 40:
            data_short_lines.append(line)


    # create dict for splitting
    data_dict = defaultdict(list)

    for line in data_short_lines:
        data_dict[line[0]+line[1]].append(line)

    # create training set
    training_pages = 5000 #total: 6572

    training_keys = random.sample(list(data_dict), training_pages)

    training_set = []

    for training_key in training_keys:
        page_data = data_dict[training_key]
        for line in page_data:
            training_set.append(line)

    # create testing set
    testing_keys = [key for key in data_dict.keys() if key not in training_keys]

    testing_set = []

    for testing_key in testing_keys:
        page_data = data_dict[testing_key]
        for line in page_data:
            testing_set.append(line)

    # remove some correct lines (i.e. increase proportion of errors)
    #drop_probability = 0.0

    #training_set_copy = training_set.copy()
    #training_set = []
    #testing_set_copy = testing_set.copy()
    #testing_set = []

    #for alignment in training_set_copy:
    #    if alignment[5] == 0.0:
    #        if random.random() > drop_probability:
    #            training_set.append(alignment)
    #    else:
    #        training_set.append(alignment)

    #for alignment in testing_set_copy:
    #    if alignment[5] == 0.0:
    #        if random.random() > drop_probability:
    #            testing_set.append(alignment)
    #    else:
    #        testing_set.append(alignment)

    # remove high CERs

    no_cer_training = 0
    tiny_cer_training = 0
    low_cer_training = 0
    medium_cer_training = 0
    high_cer_training = 0
    huge_cer_training = 0
    enormous_cer_training = 0

    no_cer_training_set = []
    tiny_cer_training_set = []
    low_cer_training_set = []
    medium_cer_training_set = []
    high_cer_training_set = []
    huge_cer_training_set = []
    enormous_cer_training_set = []

    for a in training_set:
        if a[5] == 0:
            no_cer_training+=1
            no_cer_training_set.append(a)
        elif a[5] < 0.02:
            tiny_cer_training+=1
            tiny_cer_training_set.append(a)
        elif a[5] < 0.04:
            low_cer_training+=1
            low_cer_training_set.append(a)
        elif a[5] < 0.06:
            medium_cer_training+=1
            medium_cer_training_set.append(a)
        elif a[5] < 0.08:
            high_cer_training+=1
            high_cer_training_set.append(a)
        elif a[5] < 0.1:
            huge_cer_training+=1
            huge_cer_training_set.append(a)
        else:
            enormous_cer_training+=1
            enormous_cer_training_set.append(a)

    training_set_filtered = no_cer_training_set + tiny_cer_training_set + low_cer_training_set + \
            medium_cer_training_set + high_cer_training_set + huge_cer_training_set
    random.shuffle(training_set_filtered)

    no_cer_testing = 0
    tiny_cer_testing = 0
    low_cer_testing = 0
    medium_cer_testing = 0
    high_cer_testing = 0
    huge_cer_testing = 0
    enormous_cer_testing = 0

    no_cer_testing_set = []
    tiny_cer_testing_set = []
    low_cer_testing_set = []
    medium_cer_testing_set = []
    high_cer_testing_set = []
    huge_cer_testing_set = []
    enormous_cer_testing_set = []

    for a in testing_set:
        if a[5] == 0:
            no_cer_testing+=1
            no_cer_testing_set.append(a)
        elif a[5] < 0.02:
            tiny_cer_testing+=1
            tiny_cer_testing_set.append(a)
        elif a[5] < 0.04:
            low_cer_testing+=1
            low_cer_testing_set.append(a)
        elif a[5] < 0.06:
            medium_cer_testing+=1
            medium_cer_testing_set.append(a)
        elif a[5] < 0.08:
            high_cer_testing+=1
            high_cer_testing_set.append(a)
        elif a[5] < 0.1:
            huge_cer_testing+=1
            huge_cer_testing_set.append(a)
        else:
            enormous_cer_testing+=1
            enormous_cer_testing_set.append(a)

    random.shuffle(no_cer_testing_set)
    random.shuffle(tiny_cer_testing_set)
    random.shuffle(low_cer_testing_set)
    random.shuffle(medium_cer_testing_set)
    random.shuffle(high_cer_testing_set)
    random.shuffle(huge_cer_testing_set)


    validation_set_filtered = (no_cer_testing_set[:int(len(no_cer_testing_set)/2)] + \
                                tiny_cer_testing_set[:int(len(tiny_cer_testing_set)/2)] + \
                                low_cer_testing_set[:int(len(low_cer_testing_set)/2)] + \
                                medium_cer_testing_set[:int(len(medium_cer_testing_set)/2)] + \
                                high_cer_testing_set[:int(len(high_cer_testing_set)/2)] + \
                                huge_cer_testing_set[:int(len(huge_cer_testing_set)/2)])

    testing_set_filtered = (no_cer_testing_set[int(len(no_cer_testing_set)/2):] + \
                            tiny_cer_testing_set[int(len(tiny_cer_testing_set)/2):] + \
                            low_cer_testing_set[int(len(low_cer_testing_set)/2):] + \
                            medium_cer_testing_set[int(len(medium_cer_testing_set)/2):] + \
                            high_cer_testing_set[int(len(high_cer_testing_set)/2):] + \
                            huge_cer_testing_set[int(len(huge_cer_testing_set)/2):])
    random.shuffle(testing_set_filtered)
    random.shuffle(validation_set_filtered)

    save_alignments_to_sqlite(training_set_filtered, path=training_dir, append=False)
    save_alignments_to_sqlite(testing_set_filtered, path=testing_dir, append=False)
    save_alignments_to_sqlite(validation_set_filtered, path=validation_dir, append=False)

################################################################################
@click.command()
@click.argument('in-dir', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True))
#@click.option('training-proportion', default=0.8, help='Training data proportion')
@click.option('--seed', default=49, help='')
def split_dataset_sliding_window_2(in_dir, out_dir, seed):
    '''
    Arguments:
    in-dir -- Input database
    out-dir -- Path to output databases

    NOT TESTED! DO NOT USE! (formerly run_dataset_splitting_sliding_window_charge2.py)
    '''

    random.seed(49)

    input_path = home_dir + '/Qurator/used_data/preproc_data/dta/aligned_corpus_sliding_window_german_charge2_080920.db'
    training_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_sliding_window_german_biased_charge2_170920.db'
    testing_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_sliding_window_german_biased_2charges_170920.db'
    validation_path = home_dir + '/Qurator/used_data/preproc_data/dta/validation_set_sliding_window_german_biased_2charges_small_170920.db'
    testing_small_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_sliding_window_german_biased_2charges_small_170920.db'

    german_data, german_data_as_df, headers = load_alignments_from_sqlite(path=input_path, size='total')

    combine_with_charge1 = True

    if combine_with_charge1:
        #training_charge1_path = home_dir + '/Qurator/used_data/preproc_data/dta/training_set_00-10_sliding_window_german_150620.db'
        #training_charge1_data, training_charge1_data_as_df, headers_charge1 = load_alignments_from_sqlite(path=training_charge1_path, size='total')

        #incorrect_charge1_data = []
        #for line in training_charge1_data:
        #    if line[5] > 0:
        #        incorrect_charge1_data.append(line)

        testing_charge1_path = home_dir + '/Qurator/used_data/preproc_data/dta/testing_set_00-10_sliding_window_german_150620.db'
        testing_charge1_data, testing_charge1_data_as_df, headers_charge1 = load_alignments_from_sqlite(path=testing_charge1_path, size='total')


    # remove three word lines
    german_data_compact = []

    for line in german_data:
        if len(line[4].split(' ')) == 4:
            german_data_compact.append(line)

    # remove long lines

    german_data_short_lines = []

    for line in german_data_compact:
        if len(line[4]) <= 40:
            german_data_short_lines.append(line)


    # create dict for splitting
    german_data_dict = defaultdict(list)

    for line in german_data_short_lines:
        german_data_dict[line[0]+line[1]].append(line)


    # create training set
    training_pages = 5000 #total: 5654

    training_keys = random.sample(list(german_data_dict), training_pages)

    training_set = []

    for training_key in training_keys:
        page_data = german_data_dict[training_key]
        for line in page_data:
            training_set.append(line)

    correct_lines = 0

    for line in training_set:
        if line[5] == 0:
            correct_lines += 1

    incorrect_lines = len(training_set) - correct_lines

    # create testing set
    testing_keys = [key for key in german_data_dict.keys() if key not in training_keys]

    testing_set = []

    for testing_key in testing_keys:
        page_data = german_data_dict[testing_key]
        for line in page_data:
            testing_set.append(line)


    no_cer_training = 0
    tiny_cer_training = 0
    low_cer_training = 0
    medium_cer_training = 0
    high_cer_training = 0
    huge_cer_training = 0
    enormous_cer_training = 0

    no_cer_training_set = []
    tiny_cer_training_set = []
    low_cer_training_set = []
    medium_cer_training_set = []
    high_cer_training_set = []
    huge_cer_training_set = []
    enormous_cer_training_set = []

    for a in training_set:
        if a[5] == 0:
            no_cer_training+=1
            no_cer_training_set.append(a)
        elif a[5] < 0.02:
            tiny_cer_training+=1
            tiny_cer_training_set.append(a)
        elif a[5] < 0.04:
            low_cer_training+=1
            low_cer_training_set.append(a)
        elif a[5] < 0.06:
            medium_cer_training+=1
            medium_cer_training_set.append(a)
        elif a[5] < 0.08:
            high_cer_training+=1
            high_cer_training_set.append(a)
        elif a[5] < 0.1:
            huge_cer_training+=1
            huge_cer_training_set.append(a)
        else:
            enormous_cer_training+=1
            enormous_cer_training_set.append(a)

    incorrect_training_set_filtered = tiny_cer_training_set + low_cer_training_set + \
            medium_cer_training_set + high_cer_training_set + huge_cer_training_set

    #combine charge2 training set with charge1 training set
    #if combine_with_charge1:
    #    incorrect_training_set_filtered.extend(incorrect_charge1_data)

    ten_percent = 19600
    reduced_no_cer_training_set = random.sample(no_cer_training_set, ten_percent)

    training_set_final = incorrect_training_set_filtered + reduced_no_cer_training_set
    random.shuffle(incorrect_training_set_filtered)


    # create test set

    no_cer_testing = 0
    tiny_cer_testing = 0
    low_cer_testing = 0
    medium_cer_testing = 0
    high_cer_testing = 0
    huge_cer_testing = 0
    enormous_cer_testing = 0

    no_cer_testing_set = []
    tiny_cer_testing_set = []
    low_cer_testing_set = []
    medium_cer_testing_set = []
    high_cer_testing_set = []
    huge_cer_testing_set = []
    enormous_cer_testing_set = []

    for a in testing_set:
        if a[5] == 0:
            no_cer_testing+=1
            no_cer_testing_set.append(a)
        elif a[5] < 0.02:
            tiny_cer_testing+=1
            tiny_cer_testing_set.append(a)
        elif a[5] < 0.04:
            low_cer_testing+=1
            low_cer_testing_set.append(a)
        elif a[5] < 0.06:
            medium_cer_testing+=1
            medium_cer_testing_set.append(a)
        elif a[5] < 0.08:
            high_cer_testing+=1
            high_cer_testing_set.append(a)
        elif a[5] < 0.1:
            huge_cer_testing+=1
            huge_cer_testing_set.append(a)
        else:
            enormous_cer_testing+=1
            enormous_cer_testing_set.append(a)

    testing_set_filtered = no_cer_testing_set + tiny_cer_testing_set + low_cer_testing_set + \
        medium_cer_testing_set + high_cer_testing_set + huge_cer_testing_set
#    random.shuffle(testing_set_filtered)

    random.shuffle(no_cer_testing_set)
    random.shuffle(tiny_cer_testing_set)
    random.shuffle(low_cer_testing_set)
    random.shuffle(medium_cer_testing_set)
    random.shuffle(high_cer_testing_set)
    random.shuffle(huge_cer_testing_set)

    no_cer_set_validation = no_cer_testing_set[:int(len(no_cer_testing_set)/2)]
    tiny_cer_set_validation = tiny_cer_testing_set[:int(len(tiny_cer_testing_set)/2)]
    low_cer_set_validation = low_cer_testing_set[:int(len(low_cer_testing_set)/2)]
    medium_cer_set_validation = medium_cer_testing_set[:int(len(medium_cer_testing_set)/2)]
    high_cer_set_validation = high_cer_testing_set[:int(len(high_cer_testing_set)/2)]
    huge_cer_set_validation = huge_cer_testing_set[:int(len(huge_cer_testing_set)/2)]

    #validation_size = int(len(testing_set_filtered)/2)

    #validation_set_final = testing_set_filtered[:validation_size]
    validation_set_incorrect = tiny_cer_set_validation + low_cer_set_validation + medium_cer_set_validation + high_cer_set_validation + huge_cer_set_validation
    validation_no_cer_size = 630

    validation_no_cer_sample = random.sample(no_cer_set_validation, validation_no_cer_size)
    validation_set_incorrect.extend(validation_no_cer_sample)
    validation_set_final = validation_set_incorrect.copy()

    no_cer_set_testing = no_cer_testing_set[int(len(no_cer_testing_set)/2):]
    tiny_cer_set_testing = tiny_cer_testing_set[int(len(tiny_cer_testing_set)/2):]
    low_cer_set_testing = low_cer_testing_set[int(len(low_cer_testing_set)/2):]
    medium_cer_set_testing = medium_cer_testing_set[int(len(medium_cer_testing_set)/2):]
    high_cer_set_testing = high_cer_testing_set[int(len(high_cer_testing_set)/2):]
    huge_cer_set_testing = huge_cer_testing_set[int(len(huge_cer_testing_set)/2):]

    testing_set_final = no_cer_set_testing + tiny_cer_set_testing + low_cer_set_testing + medium_cer_set_testing + high_cer_set_testing + huge_cer_set_testing

    testing_set_incorrect = tiny_cer_set_testing + low_cer_set_testing + medium_cer_set_testing + high_cer_set_testing + huge_cer_set_testing
    testing_no_cer_size = 620

    testing_no_cer_sample = random.sample(no_cer_set_testing, testing_no_cer_size)
    testing_set_incorrect.extend(testing_no_cer_sample)
    testing_set_final_small = testing_set_incorrect.copy()

    if combine_with_charge1:

        no_cer_testing_charge1 = 0
        tiny_cer_testing_charge1 = 0
        low_cer_testing_charge1 = 0
        medium_cer_testing_charge1 = 0
        high_cer_testing_charge1 = 0
        huge_cer_testing_charge1 = 0
        enormous_cer_testing_charge1 = 0

        no_cer_testing_set_charge1 = []
        tiny_cer_testing_set_charge1 = []
        low_cer_testing_set_charge1 = []
        medium_cer_testing_set_charge1 = []
        high_cer_testing_set_charge1 = []
        huge_cer_testing_set_charge1 = []
        enormous_cer_testing_set_charge1 = []

        for a in testing_charge1_data:
            if a[5] == 0:
                no_cer_testing_charge1+=1
                no_cer_testing_set_charge1.append(a)
            elif a[5] < 0.02:
                tiny_cer_testing_charge1+=1
                tiny_cer_testing_set_charge1.append(a)
            elif a[5] < 0.04:
                low_cer_testing_charge1+=1
                low_cer_testing_set_charge1.append(a)
            elif a[5] < 0.06:
                medium_cer_testing_charge1+=1
                medium_cer_testing_set_charge1.append(a)
            elif a[5] < 0.08:
                high_cer_testing_charge1+=1
                high_cer_testing_set_charge1.append(a)
            elif a[5] < 0.1:
                huge_cer_testing_charge1+=1
                huge_cer_testing_set_charge1.append(a)
            else:
                enormous_cer_testing_charge1+=1
                enormous_cer_testing_set_charge1.append(a)

        random.shuffle(no_cer_testing_set_charge1)
        random.shuffle(tiny_cer_testing_set_charge1)
        random.shuffle(low_cer_testing_set_charge1)
        random.shuffle(medium_cer_testing_set_charge1)
        random.shuffle(high_cer_testing_set_charge1)
        random.shuffle(huge_cer_testing_set_charge1)

        no_cer_set_validation_charge1 = no_cer_testing_set_charge1[:int(len(no_cer_testing_set_charge1)/2)]
        tiny_cer_set_validation_charge1 = tiny_cer_testing_set_charge1[:int(len(tiny_cer_testing_set_charge1)/2)]
        low_cer_set_validation_charge1 = low_cer_testing_set_charge1[:int(len(low_cer_testing_set_charge1)/2)]
        medium_cer_set_validation_charge1 = medium_cer_testing_set_charge1[:int(len(medium_cer_testing_set_charge1)/2)]
        high_cer_set_validation_charge1 = high_cer_testing_set_charge1[:int(len(high_cer_testing_set_charge1)/2)]
        huge_cer_set_validation_charge1 = huge_cer_testing_set_charge1[:int(len(huge_cer_testing_set_charge1)/2)]

        validation_set_incorrect_charge1 = tiny_cer_set_validation_charge1 + low_cer_set_validation_charge1 + medium_cer_set_validation_charge1 + high_cer_set_validation_charge1 + huge_cer_set_validation_charge1
        validation_no_cer_size_charge1 = 1600

        validation_no_cer_sample_charge1 = random.sample(no_cer_set_validation_charge1, validation_no_cer_size_charge1)
        validation_set_incorrect_charge1.extend(validation_no_cer_sample_charge1)
        validation_set_final.extend(validation_set_incorrect_charge1.copy())

        no_cer_set_testing_charge1 = no_cer_testing_set_charge1[int(len(no_cer_testing_set_charge1)/2):]
        tiny_cer_set_testing_charge1 = tiny_cer_testing_set_charge1[int(len(tiny_cer_testing_set_charge1)/2):]
        low_cer_set_testing_charge1 = low_cer_testing_set_charge1[int(len(low_cer_testing_set_charge1)/2):]
        medium_cer_set_testing_charge1 = medium_cer_testing_set_charge1[int(len(medium_cer_testing_set_charge1)/2):]
        high_cer_set_testing_charge1 = high_cer_testing_set_charge1[int(len(high_cer_testing_set_charge1)/2):]
        huge_cer_set_testing_charge1 = huge_cer_testing_set_charge1[int(len(huge_cer_testing_set_charge1)/2):]

        testing_set_final_charge1 = no_cer_set_testing_charge1 + tiny_cer_set_testing_charge1 + low_cer_set_testing_charge1 + medium_cer_set_testing_charge1 + high_cer_set_testing_charge1 + huge_cer_set_testing_charge1
        testing_set_final.extend(testing_set_final_charge1)

        testing_set_incorrect_charge1 = tiny_cer_set_testing_charge1 + low_cer_set_testing_charge1 + medium_cer_set_testing_charge1 + high_cer_set_testing_charge1 + huge_cer_set_testing_charge1
        testing_no_cer_size_charge1 = 1600

        testing_no_cer_sample_charge1 = random.sample(no_cer_set_testing_charge1, testing_no_cer_size_charge1)
        testing_set_incorrect_charge1.extend(testing_no_cer_sample_charge1)
        testing_set_final_small.extend(testing_set_incorrect_charge1.copy())

    random.shuffle(training_set_final)
    random.shuffle(testing_set_final)
    random.shuffle(validation_set_final)
    random.shuffle(testing_set_final_small)

    save_alignments_to_sqlite(training_set_final, path=training_path, append=False)
    save_alignments_to_sqlite(testing_set_final, path=testing_path, append=False)
    save_alignments_to_sqlite(validation_set_final, path=validation_path, append=False)
    save_alignments_to_sqlite(testing_set_final_small, path=testing_small_path, append=False)
