import pytest
import numpy as np
from datasist.nlp import convert_lower_case, remove_stopwords, remove_punctuation, remove_apostrophe, stemming


def test_lower_case():
    text = "DO YOU SHIP TO CANADA?"
    expected = np.array('do you ship to canada?', dtype='<U22')
    output = convert_lower_case(text)

    assert output == expected


def test_remove_stopwords():
    text = "I am a man"
    expected = " man"
    output = remove_stopwords(text)

    assert output == expected


def test_remove_punctuation():
    text = "Hey, I really like this!!!"
    expected = 'Hey I really like this'
    output = remove_punctuation(text)

    assert output == expected


def test_remove_apostrophe():
    text = "This is Daisy's book"
    expected = np.array('This is Daisys book', dtype='<U19')
    output = remove_apostrophe(text)

    assert output == expected


def test_stemming():
    text = "I am loving this"
    expected = ' I am love thi'
    output = stemming(text)

    assert output == expected
