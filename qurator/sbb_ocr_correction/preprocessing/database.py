import os
import pandas as pd
import sqlite3


def save_alignments_to_sqlite(alignments, path, append=True):
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

    for line in alignments:
        add_line(conn, line)

    conn.commit()


def load_alignments_from_sqlite(path, size='total', table='alignments'):
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

    if not size == 'total':
        rows_as_df = pd.read_sql_query('SELECT * from ' + table + ' LIMIT {};'.format(size), conn)
    else:
        rows_as_df = pd.read_sql_query('SELECT * from ' + table + ';', conn)

    return rows, rows_as_df, rows_as_df.columns
