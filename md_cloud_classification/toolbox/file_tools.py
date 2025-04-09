'''
Created on 22.01.2021

@author: B.Lauster
'''
import os
from collections import defaultdict


def get_filelist(basepath, recursive=True, must_contain=[], not_contain=[]):
    '''
    Get a recursive list of files sorted by filename.

    @basepath is an absolute path that contains all files.
    @recursive [default: True] enables the search in subfolders.
    @must_contain [default: empty list] is a list of strings that must be
                  in the filename
    @not_contain [default: empty list] is a list of strings that must not be
                  in the filename
    '''
    cur_filelist = os.listdir(basepath)
    final_list = []
    for cur_file in cur_filelist:
        abs_file = os.path.join(basepath, cur_file)
        if os.path.isdir(abs_file) and recursive:
            final_list.extend(get_filelist(abs_file, recursive=recursive,
                                           must_contain=must_contain,
                                           not_contain=not_contain))
        if os.path.isfile(abs_file):
            invalid = False
            for search_elem in must_contain:
                if search_elem not in cur_file:
                    invalid = True
                    break
            for search_elem in not_contain:
                if search_elem in cur_file:
                    invalid = True
                    break
            if invalid:
                continue
            final_list.append(abs_file)
    final_list.sort()
    return final_list


def filelist_to_str(filelist: list, grouping: bool=False) -> str:
    '''
    Convert a list of filesnames into a nicely formatted string

    @filelist
    @grouping (default=False) if set true, the string will be ordered by
        directory and only displays the relative filename within a given
        directory. Otherwise the output is just a line-separated list of
        all filenames
    '''
    if not grouping:
        return '\n'.join(filelist)
    # Use a dictionary to sort files to
    directories = defaultdict(list)

    # Group all files by their folder
    for filename in filelist:
        directory = os.path.dirname(filename)
        directories[directory].append(filename)

    # Create string based on ordered folders
    out = []
    for directory, filenames in directories.items():
        out.append(f"{directory}")
        for filename in filenames:
            out.append(f".{os.sep}{os.path.basename(filename)}")
        out.append("")
    return '\n'.join(out)
