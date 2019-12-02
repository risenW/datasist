'''
This module contains all functions relating to the natural language preprocessing
'''

from nltk.corpus import stopwords
import re


def clean_text(text):
    '''

    :param text: a string
    :return: a modified version of the initial string
    '''

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text
