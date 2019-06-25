# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:48:01 2019

@author: Yan
"""

# follows looesely https://medium.com/coinmonks/language-prediction-using-deep-neural-networks-42eb131444a5

import wikipedia as wiki
import string
import unicodedata
import numpy as np


def get_words(dict, remove_duplicates = True):
    ''' Search articles on wikipedia and output words in the article. Words are cleaned
    of punctuation, numbers and accents.
    Input: dict {language: [word_to_search, word_to_search,...],...}
    Output: dict {language: [word_in_article, word_in_article,...],...} '''
    new_dict = {}
    for language, queries in dict.items():
        wiki.set_lang(language)
        words = []
        for query in queries:
            text = wiki.WikipediaPage(title = query).content # extract text from wiki page
            text_words = text.split() # split the text into words
            text_words_clean = map(clean_word, text_words) # clean every word
            words += text_words_clean # add all the extracted words to the list
        if remove_duplicates:
            new_dict[language] = list(set(words)) # remove duplicate words in a given language
        else:
            new_dict[language] = words # don't remove duplicate words
    return new_dict


def clean_word(word):
    ''' Clean a given word by putting it in lower case, removing punctuation, numbers and accents.
    Input: str word
    Output str cleaned_word '''
    word = word.lower() # convert everything to lowercase
    word = word.translate(str.maketrans('', '', string.punctuation)) # remove any punctuation
    word = word.translate(str.maketrans('', '', string.digits)) # remove any number
    word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode('utf-8') # remove accents
    return word


def word_to_array(word, max_len = 12):
    ''' Convert word to numpy array, each column is a letter written in one-hot encoding. It only takes
    the first max_len letters of the word.
    Inputs: str word
            int max_len (optional)
    Output: array of size (26,max_len) encoded_word '''
    word_array = np.zeros((26,max_len))
    for i, char in enumerate(word):
        char_num = ord(char) - 97
        if char_num >= 0 and char_num <= 26 and i < max_len:
            word_array[char_num, i] = 1
    return word_array


def create_data(query_dict, max_len = 12, min_len = 3, remove_duplicates = True):
    ''' Create data for words in different languages in form of numpy array to feed to a NN.
    It only keeps the words of length in (min_len,max_len) from the wikipedia pages and cleans them.
    Input: dict {language: [word_to_search, word_to_search,...],...}
            int max_len (optional)
            int min_len (optional)
    Outputs: array of size (num_words, 26, max_len) containing all the one-hot encoded words
            array of size (num_words, num_languages) containing the one-hot encoded language for each word '''
    word_dict = get_words(query_dict, remove_duplicates)
    features = []
    labels = []
    for i, (language, word_list) in enumerate(word_dict.items()):
        for word in word_list:
            if len(word) >= min_len and len(word) <= max_len:
                features.append(word_to_array(word, max_len))
                label = np.zeros(len(word_dict))
                label[i] = 1
                labels.append(label)
    features = np.stack(features)
    labels = np.stack(labels)
    return features, labels


def array_to_word(arr):
    ''' Convert one-hot encoded array for a word to a string word.
    Inputs: array of size (26,max_len) one-hot encoded word '''
    word = ''
    word_len = arr.shape[1]
    for j in range(word_len):
        for i in range(26):
            if arr[i,j] == 1:
                word += chr(97 + i)
    return word


def count_words(data, labels):
    ''' Count number of words of each language in a dataset of words. '''
    for i in range(labels.shape[1]):
        num = np.sum(labels[:,i] == 1)
        print('Number of words in language ', i, ': ', num)
    print('Total number of words: ', data.shape[0])