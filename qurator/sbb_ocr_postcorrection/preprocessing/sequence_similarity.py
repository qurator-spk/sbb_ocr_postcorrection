import qurator.dinglehopper.edit_distance as edit_distance
import qurator.dinglehopper.character_error_rate as character_error_rate

def check_sequence_similarity(aligned_sequence, similarity_range=(0.01, 0.1), corpus_lang='de'):
    '''
    Check similarity of sequences.

    Makes use of a threshold for the maximal similarity distance. Every value
    above that threshold will be ignored.

    Keyword arguments:
    aligned_sequence (iterator) -- the sequence alignments
    threshold (float) -- the maximal similarity distance threshold
                         (default: 0.1)

    Outputs:
    ocr_sequences (list) -- the OCR sequences
    gt_sequences (list) -- the GT sequences
    max_distances (list) -- the maximally allowed distances
    similarity_encoding (list) -- the similarity values (1 or 0)
    '''

    class SimilarString:
        '''A class to check sequence similarity using a distance threshold.'''

        def __init__(self, string):
            self._string = string
            self.max_distance = max_distance

        def __eq__(self, other):
            return edit_distance.distance(self._string, other._string) <= self.max_distance    # XXX NOT the final version

        def __ne__(self, other):
            return not self.__eq__(other)

        def __repr__(self):
            return 'SimilarString(\'%s\')' % self._string

    #min_threshold = int(round((len(self._string) * max_threshold), 0))
    #max_distance = int(round((len(self._string) * similarity_range[1]), 0))
    ocr_sequences = []
    gt_sequences = []
    character_error_rates = []
    levenshtein_distances = []
    min_distances = []
    max_distances = []
    similarity_encoding = []
    left, right = zip(*list(aligned_sequence))

    #import pdb; pdb.set_trace()

    for ocr, gt in zip(left, right):

        ocr_id = ocr[0]
        ocr_seq = ocr[1]
        gt_id = gt[0]
        gt_seq = gt[1]

        try:

            min_distance = int(round((len(gt_seq) * similarity_range[0]), 0))
            max_distance = int(round((len(gt_seq) * similarity_range[1]), 0))
            levenshtein_distance = edit_distance.distance(ocr_seq, gt_seq)
            cer = character_error_rate(gt_seq, ocr_seq)

            if min_distance <= levenshtein_distance <= max_distance:
                ocr_sequences.append((ocr_id, ocr_seq))
                gt_sequences.append((gt_id, gt_seq))
                try:
                    #max_distance = SimilarString(gt).max_distance

                    min_distances.append(min_distance)
                    max_distances.append(max_distance)
                    levenshtein_distances.append(levenshtein_distance)
                    character_error_rates.append(cer)

                    if SimilarString(ocr_seq) == SimilarString(gt_seq):
                        similarity_encoding.append(1)
                    else:
                        similarity_encoding.append(0)
                except TypeError:
                    levenshtein_distances.append(-1)
                    max_distances.append(-1)
                    similarity_encoding.append(0)
                    continue

        except Exception as e:
            #print(type(e))
            #print(e)
            pass

    levenshtein_distances = list(map(int, levenshtein_distances))
    character_error_rates = list(map(float, character_error_rates))

    return ocr_sequences, gt_sequences, character_error_rates, levenshtein_distances, min_distances, max_distances, similarity_encoding


def print_alignment_stats(idx, length, num_similar_sequences, scope='PAGE'):
    '''
    Prints basic alignment related stats.

    Keyword arguments:
    idx (str) -- the ID
    length (int) -- the number of sequences
    num_similar_sequences (int) -- the number of similar sequences
    scope (str) -- the scope of the stats (e.g. PAGE, DOC) (default: 'PAGE')

    '''
    print('\n{} ID: {}'.format(scope, idx))
    print('TOTAL SEQUENCES: {}'.format(length))
    print('ALIGNED SEQUENCES: {}'.format(num_similar_sequences))
    try:
        print('ALIGNMENT PROPORTION: {:.2f}'.format(num_similar_sequences/length))
    except:
        pass

def print_distance_summary(seq1, seq2, max_distance, similarity_encoding, line_id):
    '''
    Print distance summary.

    Summary metrics are: Line ID, Levenshtein distance, maximal distance,
                         similarity encoding

    Keyword arguments:
    seq1 (str): the first string
    seq2 (str): the second string
    max_distance (int): the maximally allowed distance
    similarity_encoding (int): the similarity value (1 or 0)
    line_id (int): the line ID
    '''

    try:
        print('Line: %03d; Levenshtein: %d; Allowed Dist: %d, Similarity: %d' % (line_id, edit_distance.distance(seq1, seq2), max_distance, similarity_encoding))
    except TypeError:
        if seq1 is None or seq2 is None:
            print('Line: %03d; NoneType seq found' % (line_id))
        else:
            print('Line: %03d; Unknown Error' % (line_id))
