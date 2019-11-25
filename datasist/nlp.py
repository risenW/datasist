'''
This module contains all functions relating to nlp
'''
import re
import os
import spacy 
os.system("python -m spacy download en")
nlp_m = spacy.load('en')
from spacy.lang.en.stop_words import STOP_WORDS


# other_puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
#  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
#  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
#  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
#  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]




def pre_process(text=None, stopword=True, punctuation=False, number=True):
    '''
    Clean unstructured free text or corpus input for Natural Language Processing computing task at a go.
    Parameter:
    ----------------------------
        text: Free text assigned in a variable.
            Text formart should be: x = u" Free Text"
        stopword: bool, Default True
            Words that do not provide any useful information to decide in which category a text should be classified.
        punctuation: bool, Default False
            Unique and conventional characters that aid understanding of sentences which need to be removed during pre-processing or left for easy text chunking or tokenization.
        Number: bool, Default True
            Reepresent numbers as special characters.
    Returns:
    ----------------------------
            Corpus: Pre-processed corpus without stopwords or punctuations or both
    '''
    if text is None:
        raise ValueError("text: Expecting a Corpus/ Free Text, got 'None'")
    
    sent = nlp_m(text)
    filtr = []
    filtr_ = []
    punctuations = '''!()[]{};:,'"<>./?@+#$%^&*_~'''
    no_punct = " "

    # remove stopwords from corpus
    for i in sent:
        if i.is_stop ==False:
            filtr.append(i)


    # remove punctuation from the string
    for char in str(filtr):
        if char not in punctuations:
            no_punct = no_punct + char
    word = no_punct.split("\n")

    # replace digits with #####
    if bool(re.search(r'\d', str(word))):
        word = re.sub('[0-9]{5,}', '#####', str(word))
        word = re.sub('[0-9]{4}', '####', str(word))
        word = re.sub('[0-9]{3}', '###', str(word))
        word = re.sub('[0-9]{2}', '##', str(word))
    
    # remove stopword without removal of punctuatuion for easy tokenization
    if not punctuation:
        for word_ in sent:
            if word_.is_stop == False:
                filtr_.append(word_)
        # replace digits with #####
        if bool(re.search(r'\d', str(filtr_))):
            punct_word = re.sub('[0-9]{5,}', '#####', str(filtr_))
            punct_word = re.sub('[0-9]{4}', '####', str(punct_word))
            punct_word = re.sub('[0-9]{3}', '###', str(punct_word))
            punct_word = re.sub('[0-9]{2}', '##', str(punct_word))
        return punct_word
    else:
        return word