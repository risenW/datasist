import pytest
import numpy as np
from datasist.nlp import convert_lower_case, remove_stopwords, remove_punctuation, remove_apostrophe, stemming, sentence_similarity, remove_newline, remove_unwanted_chars


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

def test_sentence_similarity():
    sentence1 = "Datasist is a very amazing package"
    sentence2 = "I think Datasist is a very amazing package"
    expected = 0.866
    output = sentence_similarity(sentence1, sentence2)

    assert output == expected

def test_remove_newline():
    text = ")This is a line \n This is a new line ^ -- * 2020"
    expected = "This is a line This is a new line 2020"
    output = remove_newline(text)

    assert output == expected 

def test_remove_unwanted_chars():
    text = "Me &amp; The Big Homie meanboy3000 #MEANBOY #MB #MBS #MMR #STEGMANLIFE @Stegman St. <url>	"
    expected = "Me amp The Big Homie meanboy MEANBOY MB MBS MMR STEGMANLIFE Stegman St url"
    output = remove_unwanted_chars(text)

    assert output == expected


