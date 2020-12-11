import os
import string


def get_file_paths(dir_name):
    '''
    Extract all files in dir_name (and sub dirs).
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


def create_char_set(extra_chars=None):
    if not extra_chars:
        return string.printable
    else:
        return string.printable + extra_chars
