import subprocess
import tarfile


def get_data_from_git_annex(path):
    '''
    Retrieve data from git annex.

    Keyword arguments:
    path (str) -- the path of the git annex link.
    '''

    subprocess.run(['git', 'annex', 'get', path])


def delete_unnecessary_file_links(path):
    '''
    Remove working git annex link.

    Keyword arguments:
    path (str) -- the path of the git annex link
    '''

    subprocess.call(['git', 'annex', 'drop', path])


def extract_tar(path):
    '''
    Extract tar file.

    Keyword arguments:
    path (str) -- the tar file path
    '''

    tf = tarfile.open(path)

    tf.extractall()


def split_list_in_n_sized_chunks(l, n):
    '''
    Split a list in n (ideally) equal chunks.

    If the list cannot be split into equal chunks, the last chunk will contain
    the remaining items.

    Source: https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/

    Keyword arguments:
    l (list): the list
    n (int): the size of a chunk

    Outputs:
    the chunk iterator
    '''

    for i in range(0, len(l), n):
        yield l[i:i+n]
