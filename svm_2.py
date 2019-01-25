import sys
import dill as pickle
import numpy as np
from itertools import chain
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold
import xgboost as xgb
from sklearn import svm
from collections import Counter
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from score import *
import score
import joblib
from sklearn.metrics import confusion_matrix


'''
This function reads the training data file, converts the stances to categorical 
variables, takes 60% of the training data for cross validation, and generates features 
using the FeatureGenerator files.
'''
def build_data():

    # read training data file
    data = pd.read_csv('./data/merged_data_train.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    data = data[used_column].dropna()
    data['Headline'] = data['claimHeadline'].apply(lambda x: x[8:])
    data['articleBody'] = data['articleHeadline']
    data['Body ID'] = data['articleId']
    targets =['unknown', 'false', 'true']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = list(map(lambda x: targets_dict[x], data['claimTruthiness']))

    train = data.sample(frac=0.6, random_state=2018)
    test = data.loc[~data.index.isin(train.index)]

    data_y = train['target'].values

    # generate features
    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()
                 ]
    features = [f for g in generators for f in g.read('train')]

    # print data shapes
    data_x = (np.hstack(features))
    print(data_x[0,:])
    print('data_x.shape')
    print(data_x.shape)
    print('data_y.shape')
    print(data_y.shape)
    print('body_ids.shape')
    print(data['Body ID'].values.shape)


    return data_x, data_y, data['Body ID'].values, test[['target', 'Headline', 'Body ID']]

'''
Basically the same as build_data, but for test data
'''
def build_test_data():

    # create target variable
    # replace file names when test data is ready
    data = pd.read_csv('./data/merged_data_train.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    data = data[used_column].dropna()
    data['Headline'] = data['claimHeadline'].apply(lambda x: x[8:])
    data['articleBody'] = data['articleHeadline']
    data['Body ID'] = data['articleId']
    targets =['unknown', 'false', 'true']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = list(map(lambda x: targets_dict[x], data['claimTruthiness']))


    train = data.sample(frac=0.6, random_state=2018)
    test = data.loc[~data.index.isin(train.index)]

    # read features
    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()
                 ]

    features = [f for g in generators for f in g.read("test")]
    print(len(features))
    #return 1

    data_x = np.hstack(features)
    print(data_x[0,:])
    print('test data_x.shape')
    print(data_x.shape)
    print('test body_ids.shape')
    print(test['Body ID'].values.shape)
                   # pair id
    return data_x, test['Body ID'].values, test['target']


def train():

    data_x, data_y, body_ids, target_stance = build_data()
    print(data_x, data_y, body_ids, target_stance)
    # read test data
    test_x, body_ids_test, true_y = build_test_data()

    print(Counter(data_y))

    # create the SVM model, generating pickle file
    bst = svm_FNC.fit(data_x, data_y)
    joblib.dump(bst, 'svm_cluster_train.pkl')

    pred_y = bst.predict(test_x)
    print(len(pred_y))
    print("------------------------------------")
    # add encoding and diplay evaluation
    df_test = pd.DataFrame()
    df_test['pred_y'] = pred_y
    df_test['true_y'] = true_y
    df_test['pred_y'] = df_test['pred_y'].replace([0,1,2], ['unknown','false', 'true'])
    df_test['true_y'] = df_test['true_y'].replace([0,1,2], ['unknown','false', 'true'])
    df_test = df_test.dropna()

    print(df_test['pred_y'].value_counts())
    print(df_test['true_y'].value_counts())

    print(score.report_score(df_test['true_y'], df_test['pred_y']))

    predicted = [LABELS[int(a)] for a in pred_y]

    stances = target_stance

    df_output = pd.DataFrame()
    df_output['Headline'] = stances['Headline']
    df_output['Body ID'] = stances['Body ID']

    df_output['Stance'] = predicted
    df_output.to_csv('tree_pred_prob_cor2.csv', index=False)
    df_output[['Headline','Body ID','Stance']].to_csv('tree_pred_cor2.csv', index=False)

    print(df_output)
    print(Counter((df_output['Stance'])))



if __name__ == '__main__':

    '''
    Here, please change the hyper-parameters for svm
    '''
    svm_FNC = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state= 2018)
    train()
