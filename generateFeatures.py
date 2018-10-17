import nltk
import pandas as pd
import numpy as np
import pickle
from helpers import *
from ngram import NGram
from nltk import ngrams
from sklearn.model_selection import train_test_split
#from CountFeatureGenerator import *
from TfidfFeatureGenerator import *

from SvdFeatureGenerator import *
#from Word2VecFeatureGenerator import *
#from SentimentFeatureGenerator import *
#from AlignmentFeatureGenerator import *

def process():

    full_data = pd.read_csv('./data/merged_data_tain.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    full_data = full_data[used_column].dropna()
    full_data['Headline'] = full_data['claimHeadline'].apply(lambda x: x[8:])
    full_data['articleBody'] = full_data['articleHeadline']
    full_data['Body ID'] = full_data['articleId']
    targets = ['observing', 'for', 'against', 'ignoring']
    targets_dict = dict(zip(targets, range(len(targets))))
    full_data['target'] = map(lambda x: targets_dict[x], full_data['articleStance'])

    train = full_data.sample(frac=0.6, random_state=2018)
    test = full_data.loc[~full_data.index.isin(train.index)]


    print('train.shape:')
    print(train.shape)
    n_train = train.shape[0]


    data = full_data
    test_flag = True
    if test_flag:

        data = full_data
        print(data)
        print('data.shape:')
        print(data.shape)

        train = train
        print(train)
        print('train.shape:')
        print(train.shape)

        test = test
        print(test)
        print('test.shape:')
        print(test.shape)


    print("generate unigram")
    data["Headline_unigram"] = data["Headline"].map(lambda x: (list(nltk.word_tokenize(x))))
    data["articleBody_unigram"] = data["articleBody"].map(lambda x: (list(nltk.word_tokenize(x))))
    print(data["Headline_unigram"])

    print("generate bigram")
    data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    print(data["Headline_bigram"])

    print("generate trigram")
    join_str = "_"
    data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    print(data["Headline_trigram"])


    with open('data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)
        print('dataframe saved in data.pkl')
    #return 1

    # define feature generators
    #countFG    = CountFeatureGenerator()
    #tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    #word2vecFG = Word2VecFeatureGenerator()
    #sentiFG    = SentimentFeatureGenerator()
    #walignFG   = AlignmentFeatureGenerator()
    #generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    generators = [svdFG]
    #generators = [tfidfFG]
    #generators = [countFG]
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
