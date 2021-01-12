import io
import json
from collections import defaultdict
import os
import pandas as pd
import sqlite3

class Corpus():
    def __init__(self, pre_init_corpus=None):
        '''
        '''
        if pre_init_corpus:
            if isinstance(pre_init_corpus, dict):
                self.docs = pre_init_corpus
                self.aligned_sequences = []
            elif isinstance(pre_init_corpus, list):
                self.docs = defaultdict(defaultdict)
                self.aligned_sequences = pre_init_corpus
        else:
            self.docs = defaultdict(defaultdict)
            self.aligned_sequences = []

        if len(self.docs) > 0:
            self.doc_ids = list(self.docs.keys())
        else:
            self.doc_ids = []

    def __len__(self):
        return len(self.aligned_sequences)

    def __getitem__(self, id):
        return self.aligned_sequences[id]

    def add_doc(self, doc_id, doc):
        '''
        '''
        self.docs[doc_id] = doc
        self.doc_ids.append(doc_id)

    def convert_to_json_format(self):
        '''
        '''
        self.docs = defaultdict()

        for line in self.aligned_sequences:
            if line[0] not in self.docs.keys():
                self.docs[line[0]] = defaultdict()

            if line[1] not in self.docs[line[0]].keys():
                self.docs[line[0]][line[1]] = [[] for i in range(0,7)]

            self.docs[line[0]][line[1]][0].append([line[2], line[3]])
            self.docs[line[0]][line[1]][1].append([line[2], line[4]])

            for i in range(2,7):
                self.docs[line[0]][line[1]][i].append(line[i+3])


    def convert_to_sqlite_format(self, only_similar=True):
        '''
        Gather alignments, CER, Levenshtein and max distances, and similarity scores.

        Keyword arguments:
        aligned_corpus (dict) -- the alignment corpus
        only_similar (bool) -- defines if only similar sequence should be gathered
                            (default: True)

        Outputs
        aligned_sequences (list) -- the similar sequences
        '''
        self.aligned_sequences = []

        if self.docs:
            for doc_id, doc in self.docs.items():
                for page_id, page in doc.items():
                    for ocr, gt, cer, levenshtein, min_distance, max_distance, similarity_value in zip(page[0], page[1], page[2], page[3], page[4], page[5], page[6]):
                        ocr_id = ocr[0]
                        ocr_seq = ocr[1]
                        gt_id = gt[0]
                        gt_seq = gt[1]

                        assert ocr_id == gt_id, 'OCR and GT sequence ID is not identical.'
                        if only_similar is True:
                            if similarity_value == 1:
                                self.aligned_sequences.append((doc_id, page_id, ocr_id, ocr_seq, gt_seq, cer, int(levenshtein), min_distance, max_distance, similarity_value))
                        else:
                            self.aligned_sequences.append((doc_id, page_id, ocr_id, ocr_seq, gt_seq, cer, int(levenshtein), min_distance, max_distance, similarity_value))
        else:
            raise ValueError('No document added to corpus yet. Use .add_doc().')


    def save_to_json(self, path):
        '''
        '''
        with io.open(path, mode='w') as f_out:
            json.dump(self.docs, f_out)

    def load_from_json(self, path):
        '''
        '''
        with io.open(path, mode='r') as f_in:
            self.docs = json.load(f_in)

    def save_to_sqlite(self, path, append=True):
        '''
        Save alignments to sqlite db.

        Keyword arguments:
        alignments (list) -- the alignments + additional line content (levenshtein,
                        etc.)
        path (str) -- the path to the sqlite db
        append (bool) -- defines if lines should be appended if db exists, if False
                        a new db will be initialized (default: True)
        '''
        create_table_sql = '''CREATE TABLE IF NOT EXISTS alignments (
                                doc_id text,
                                page_id text,
                                line_id text,
                                ocr text,
                                gt text,
                                cer real,
                                levenshtein integer,
                                min_dist integer,
                                allowed_dist integer,
                                similarity integer
                            );'''

        if not len(self.aligned_sequences) > 0:
            raise ValueError('Aligned sequences not yet gathered. Use .convert_to_sqlite_format().')

        if not append:
            if os.path.isfile(path):
                os.remove(path)

        def create_connection(path):
            '''
            Create connection to sqlite db.

            Argument keywords:
            path (str) -- the path of the sqlite db

            Outputs:
            conn (obj) -- the db object
            '''

            conn = None
            try:
                conn = sqlite3.connect(path)
            except RuntimeError as e:
                print(e)
            return conn

        def create_table(conn, sql):
            '''
            Create table in sqlite db.

            Argument keywords:
            conn (obj) -- the db object
            sql (str) -- the sql query
            '''

            try:
                cur = conn.cursor()
                cur.execute(sql)
            except RuntimeError as e:
                print(e)

        def add_line(conn, line):
            '''
            Add line to db.

            Argument keywords:
            conn (obj) -- the db object
            line (set) -- the line content (alignment, etc.)
            '''
            sql = 'INSERT INTO alignments(doc_id, page_id, line_id, ocr, gt, cer, levenshtein, min_dist, allowed_dist, similarity) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'

            cur = conn.cursor()
            cur.execute(sql, line)

        conn = create_connection(path)
        create_table(conn, create_table_sql)

        for line in self.aligned_sequences:
            add_line(conn, line)

        conn.commit()

    def load_from_sqlite(self, path, size='total', table='alignments'):
        '''
        Load alignments from sqlite db.

        Keyword arguments:
        path (str) -- the path of the sqlite db
        size (int/ str) -- the number of alignments to be loaded (or 'total') (default: 'total')
        table (str) -- the name of the table in the db (default: 'alignments')

        Outputs:
        rows (list) -- the selected rows
        rows_as_df (pd.DataFrame) -- the rows stored in a pandas data frame
        rows_as_df.columns -- the column names of the pandas data frame
        '''

        def create_connection(path):
            '''
            Create connection to sqlite db.

            Argument keywords:
            path (str) -- the path of the sqlite db

            Outputs:
            conn (obj) -- the db object
            '''

            conn = None
            try:
                conn = sqlite3.connect(path)
            except RuntimeError as e:
                print(e)
            return conn

        def select_all(conn):
            '''
            Select all lines from table.

            Keyword arguments:
            conn (obj) -- the db object

            Outputs:
            rows (list) -- the selected rows
            '''
            cur = conn.cursor()

            if not size == 'total':
                cur.execute('SELECT * FROM ' + table + ' LIMIT {};'.format(size))
            else:
                cur.execute('SELECT * FROM ' + table + ';')

            rows = cur.fetchall()

            return rows

        conn = create_connection(path)
        rows = select_all(conn)

        #if not size == 'total':
        #    rows_as_df = pd.read_sql_query('SELECT * from ' + table + ' LIMIT {};'.format(size), conn)
        #else:
        #    rows_as_df = pd.read_sql_query('SELECT * from ' + table + ';', conn)

        self.aligned_sequences = rows
