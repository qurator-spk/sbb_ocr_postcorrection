import codecs
import json
import sqlite3
from sqlite3 import Error
from bs4 import BeautifulSoup


def load_dta_xml(f_paths, save_as_json=True, verbose=True):
    '''
    Load texts of the German Text Archive (Deutsches Textarchiv (DTA)).

    The texts are stored in XML (TEI).

    TEI documentation (in German):  http://www.deutschestextarchiv.de/doku/basisformat/
    '''
    dta_json = []

    def get_text_from_xml(f_path):
        '''
        Extracts text from xml.
        '''
        with codecs.open(f_path, mode='r') as tei:
            soup = BeautifulSoup(tei, 'lxml')

        def element_to_text(element):
            '''
            Checks if element exists and returns text.
            '''
            if element:
                return element.getText(separator=' ', strip=True)
            else:
                return 'NA'

        text = {
            'title': element_to_text(soup.title),
            'front': element_to_text(soup.front),
            'content': element_to_text(soup.body),
            'back': element_to_text(soup.back),
            }

        return text

    c = 0
    for f_path in f_paths:
        c += 1
        if verbose and c % 100 == 0:
            print('Processed XML files: {}'.format(c))
        dta_json.append(get_text_from_xml(f_path))

    if save_as_json:
        with codecs.open('dta_texts.json', mode='w') as f_out:
            json.dump(dta_json, f_out)

    return dta_json


def load_fulltexts(db_path, size=10000):
    '''
    Loads fulltexts created by OCR.
    '''

    def create_connection(db_path):
        '''
        '''
        conn = None
        try:
            conn = sqlite3.connect(db_path)
        except Error as e:
            print(e)
        return conn

    def select_all(conn):
        '''
        '''
        cur = conn.cursor()
        cur.execute("SELECT * FROM text LIMIT {};".format(size))

        rows = cur.fetchall()

        return rows

    def convert_to_json(rows):
        '''
        '''
        json_data = []
        for row in rows:
            json_data.append({
                'id': row[0],
                'xml_file': row[1],
                'file_id': row[2],
                'text': row[3]
            })
        return json_data

    conn = create_connection(db_path)
    rows = select_all(conn)
    return convert_to_json(rows)
