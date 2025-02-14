'''
Created on 22.01.2021

@author: B.Lauster
'''
import os


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
