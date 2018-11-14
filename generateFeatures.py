#generateFeatures.py

'''
This file serves as the driver for data preprocessing
as well as feature extraction.
'''

import nltk
import pandas as pd
import numpy as np
import dill as pickle

from helpers import *
from ngram import NGram
from nltk import ngrams
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from sklearn.model_selection import train_test_split


def print_gram(data, gram_type, print_grams):
    gram = gram_type + "gram"
    print("Sample " + gram + ": " + str(data["Headline_" + gram][0]) + "\n")
    if print_grams:
        print(data["Headline_" + gram])

# Function to generate unigrams, bigrams, and trigrams of headlines
# Set print_grams=True to print out all unigrams, bigrams, and trigrams
def generate_grams(data, print_grams=False):
    print("Generating unigrams...")
    data["Headline_unigram"] = data["Headline"].map(lambda x: (list(nltk.word_tokenize(x))))
    data["articleBody_unigram"] = data["articleBody"].map(lambda x: (list(nltk.word_tokenize(x))))
    print_gram(data, "uni", print_grams)

    print("Generating bigrams...")
    data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    print_gram(data, "bi", print_grams)

    print("Generating trigrams...")
    join_str = "_"
    data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    print_gram(data, "tri", print_grams)

# Reads the data in, builds data frame with column labels
def process_data(article_stance=True):
    full_data = pd.read_csv('./data/merged_data_tain.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    full_data = full_data[used_column].dropna()
    full_data['Headline'] = full_data['claimHeadline'].apply(lambda x: x[8:])
    full_data['articleBody'] = full_data['articleHeadline']
    full_data['Body ID'] = full_data['articleId']

    # if we want to predict articlestance or claim_Truthiness
    if article_stance:
        targets = ['observing', 'for', 'against', 'ignoring']
        targets_dict = dict(zip(targets, range(len(targets))))
        full_data['target'] = list(map(lambda x: targets_dict[x], full_data['articleStance']))
    else:
        targets = ['unknown', 'false', 'true']
        targets_dict = dict(zip(targets, range(len(targets))))
        full_data['target'] = list(map(lambda x: targets_dict[x], full_data['claimTruthiness']))

    return full_data


def process():

    full_data = process_data(article_stance=True)

    train = full_data.sample(frac=0.6, random_state=2018)
    test = full_data.loc[~full_data.index.isin(train.index)]

    print('train.shape: ' + str(train.shape))
    n_train = train.shape[0]

    data = full_data
    test_flag = False
    if test_flag:
        print('data.shape: ' + str(data.shape))
        print(data)

        print('train.shape: ' + str(train.shape))
        print(train)

        print('test.shape: ' + str(test.shape))
        print(test)

    generate_grams(data, print_grams=False)

    with open('data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)
        print('dataframe saved in data.pkl')

    # define feature generators
    countFG    = CountFeatureGenerator() # done
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator() # done
    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    ###########################################################
    #Be sure you run the tfidf again to generate the similarity
    ###########################################################
    #generators = [countFG]
    #generators = [tfidfFG]
    #generators = [sentiFG]
    #generators = [walignFG]

    for g in generators:
        g.process(data)

    for g in generators:
        g.read('train')
    for g in generators:
        g.read('test')

    print('done')

if __name__ == "__main__":
    process()
