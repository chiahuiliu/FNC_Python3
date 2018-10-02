import ngram
import pandas as pd
import numpy as np
import cPickle
from helpers import *
from sklearn.model_selection import train_test_split
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
#from AlignmentFeatureGenerator import *

def process():

    full_data = pd.read_csv('./data/merged_data_tain.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance']
    full_data = full_data[used_column]
    full_data = full_data.dropna()
    train, test = train_test_split(full_data, test_size=0.33, random_state=1234)

    read = False
    if not read:

        targets = ['observing', 'for', 'against', 'ignoring']
        targets_dict = dict(zip(targets, range(len(targets))))
        train['target'] = map(lambda x: targets_dict[x], train['articleStance'])
        print 'train.shape:'
        print train.shape
        n_train = train.shape[0]

        data = train
        # read test set, no 'Stance' column in test set -> target = NULL
        # concatenate training and test set
        test_flag = True
        if test_flag:

            data = train
            print data
            print 'data.shape:'
            print data.shape

            train = data[~data['target'].isnull()]
            print train
            print 'train.shape:'
            print train.shape

            test = data[data['target'].isnull()]
            print test
            print 'test.shape:'
            print test.shape

        #data = data.iloc[:100, :]

        #return 1

        print "generate unigram"
        data["claimHeadline_unigram"] = data["claimHeadline"].map(lambda x: preprocess_data(x))
        data["articleHeadline_unigram"] = data["articleHeadline"].map(lambda x: preprocess_data(x))

        print "generate bigram"
        join_str = "_"
        data["claimHeadline_bigram"] = data["claimHeadline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        data["articleHeadline_bigram"] = data["articleHeadline_unigram"].map(lambda x: ngram.getBigram(x, join_str))

        print "generate trigram"
        join_str = "_"
        data["claimHeadline_trigram"] = data["claimHeadline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        data["articleHeadline_trigram"] = data["articleHeadline_bigram"].map(lambda x: ngram.getTrigram(x, join_str))

        with open('data.pkl', 'wb') as outfile:
            cPickle.dump(data, outfile, -1)
            print 'dataframe saved in data.pkl'

    else:
        with open('data.pkl', 'rb') as infile:
            data = cPickle.load(infile)
            print 'data loaded'
            print 'data.shape:'
            print data.shape
    #return 1

    # define feature generators
    countFG    = CountFeatureGenerator()
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator()
    #walignFG   = AlignmentFeatureGenerator()
    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    #generators = [svdFG, word2vecFG, sentiFG]
    #generators = [tfidfFG]
    #generators = [countFG]
    #generators = [walignFG]

    for g in generators:
        g.process(data)

    for g in generators:
        g.read('train')

    #for g in generators:
    #    g.read('test')

    print 'done'


if __name__ == "__main__":

    process()

 #   Copyright 2017 Cisco Systems, Inc.
 #
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
