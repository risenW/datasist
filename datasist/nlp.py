# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# from nltk.cluster.util import cosine_distance
# import string
# import re
# import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')


# def convert_lower_case(data):
#     return np.char.lower(data)


# def remove_stopwords(data):
#     stop_words = stopwords.words('english')
#     words = word_tokenize(str(data))
#     new_text = ""
#     for w in words:
#         if w not in stop_words and len(w) > 1:
#             new_text = new_text + " " + w
#     return new_text


# def remove_punctuation(mess):
#     nopunc =[char for char in mess if char not in string.punctuation]
#     nopunc=''.join(nopunc)

#     return nopunc


# def remove_apostrophe(data):
#     return np.char.replace(data, "'", "")


# def stemming(data):
#     stemmer = PorterStemmer()

#     tokens = word_tokenize(str(data))
#     new_text = ""
#     for w in tokens:
#         new_text = new_text + " " + stemmer.stem(w)
#     return new_text


# def flatten_list(list_of_list):
#     flat_list = [item for sublist in list_of_list for item in sublist]
#     return flat_list


# def sentence_similarity(sentence1, sentence2, stopwords = nltk.corpus.stopwords.words('english')):
    
#     """ 
#     Finds the Lexical similarity between two sentences based on cosine similarity
    
#     Parameters :-
#     ------------------------------------
#           sentence1 (str) : First sentence
#           sentence2 (str) : Second sentence that is compared to the first sentence
#           stopwords (list) : List of stopwords for filtering commonly used words
          
#     Returns :-
#     -------------------------------------
#           similarity (float) : Similarity index between the two sentences represented in a probabilistic form (0.0 <= x <= 1.0)      
#     """
#     if stopwords == None:
#         stopwords = []
        
#     assert len(sentence1) > 0 and len(sentence2) > 0, "Each sentence must contain at least one word !"
    

#     sent1 = [word.lower() for word in sentence1.split(" ")]
#     sent2 = [word.lower() for word in sentence2.split(" ")]

#     all_words = list(set(sent1+sent2))
#     vect_array1 = np.zeros(len(all_words))
#     vect_array2 = np.zeros(len(all_words))

#     for word in sent1:
#         if word in stopwords:
#             continue
#         else:
#             vect_array1[all_words.index(word)] += 1

#     for word in sent2:
#         if word in stopwords:
#             continue
#         else:
#             vect_array2[all_words.index(word)] += 1
    
#     similarity = np.round(1 - cosine_distance(vect_array1, vect_array2), 3)

#     return similarity


# def remove_newline(text):
#     """
#         Remove newline character and clean text particularly extracted from documents

#         Parameters :
#         --------------------------------------------------
#         text (str) : Text to be cleaned

#         Returns :
#         --------------------------------------------------
#         sentence
#     """
#     cleaned_text = []
#     for i in text.split():
#         if i == "\n":
#             continue
#         else: 
#             cleaned_text.append(re.sub(r'[^a-zA-Z0-9_]', "", i))
    
#     return " ".join([x for x in cleaned_text if x.strip()])


# def remove_unwanted_chars(sentence):

#     """
#         Remove unwanted Characters in-order to clean raw text data in tweets and html docs particularly

#         Paramters :
#         -----------------------------------------------------
#         sentence (str)
#     """
#     mentions_pattern = r'@[A-Za-z0-9_]+'
#     http_pattern = r'http(s?)://[^ ]+'
#     www_pattern = r'www.[^ ]+'
#     non_alphabets = r'[^a-zA-Z]+'
#     combined_pattern_removal = r'|'.join((mentions_pattern, http_pattern, www_pattern, non_alphabets))
#     return re.sub(combined_pattern_removal, " ", sentence).strip()
