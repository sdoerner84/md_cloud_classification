'''
Created on Dec 19, 2024

@author: sziegler
'''


class MDCCError(Exception):
    '''
    Define generic max-doas cloud classification exception
    '''
    def __init__(self, value):
        self.value = value
        self.strerror = repr(value)

    def __str__(self):
        return self.strerror
