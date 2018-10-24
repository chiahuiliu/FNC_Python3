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
import joblib

def build_data():

    data = pd.read_csv('./data/merged_data_tain.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    data = data[used_column].dropna()
    data['Headline'] = data['claimHeadline'].apply(lambda x: x[8:])
    data['articleBody'] = data['articleHeadline']
    data['Body ID'] = data['articleId']
    targets = ['observing', 'for', 'against', 'ignoring']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = (map(lambda x: targets_dict[x], data['articleStance']))
    #data['target'] = data['target'].astype(int)

    train = data.sample(frac=0.6, random_state=2018)
    test = data.loc[~data.index.isin(train.index)]

    data_y = train['target'].values

    # read features
    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()
                  #AlignmentFeatureGenerator()
                 ]
    features = [f for g in generators for f in g.read('train')]

    data_x = (np.hstack(features))
    print(data_x[0,:])
    print('data_x.shape')
    print(data_x.shape)
    print('data_y.shape')
    print(data_y.shape)
    print('body_ids.shape')
    print(data['Body ID'].values.shape)

    #with open('data_new.pkl', 'wb') as outfile:
    #    cPickle.dump(data_x, outfile, -1)
    #    print 'data saved in data_new.pkl'

    return data_x, data_y, data['Body ID'].values

def build_test_data():

    # create target variable
    # replace file names when test data is ready
    data = pd.read_csv('./data/merged_data_tain.csv', encoding='utf-8')
    used_column = ['claimHeadline', 'articleHeadline', 'claimTruthiness', 'articleStance', 'articleId']

    data = data[used_column].dropna()
    data['Headline'] = data['claimHeadline'].apply(lambda x: x[8:])
    data['articleBody'] = data['articleHeadline']
    data['Body ID'] = data['articleId']
    targets = ['observing', 'for', 'against', 'ignoring']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = (map(lambda x: targets_dict[x], data['articleStance']))


    train = data.sample(frac=0.6, random_state=2018)
    test = data.loc[~data.index.isin(train.index)]

    '''
    body = pd.read_csv("test_bodies.csv")
    stances = pd.read_csv("test_stances_unlabeled.csv") # needs to contain pair id
    data = pd.merge(stances, body, how='left', on='Body ID')
    '''
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
    return data_x, test['Body ID'].values

def fscore(pred_y, truth_y):

    # targets = ['agree', 'disagree', 'discuss', 'unrelated']
    # y = [0, 1, 2, 3]
    score = 0
    if pred_y.shape != truth_y.shape:
        raise Exception('pred_y and truth have different shapes')
    for i in range(pred_y.shape[0]):
        if truth_y[i] == 3:
            if pred_y[i] == 3: score += 0.25
        else:
            if pred_y[i] != 3: score += 0.25
            if truth_y[i] == pred_y[i]: score += 0.75

    return score

def perfect_score(truth_y):

    score = 0
    for i in range(truth_y.shape[0]):
        if truth_y[i] == 3: score += 0.25
        else: score += 1

    return score

def eval_metric(yhat, dtrain):
    y = dtrain.get_label()
    yhat = np.argmax(yhat, axis=1)
    predicted = [LABELS[int(a)] for a in yhat]
    actual = [LABELS[int(a)] for a in y]
    s, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)
    score = float(s) / s_perf
    return 'score', score

def train():

    data_x, data_y, body_ids = build_data()
    # read test data
    test_x, body_ids_test = build_test_data()
    '''
    commenting this out as this resulted in imbalance score
    w = np.array([1 if y == 3 else 4 for y in data_y])
    print('w:')
    print(w)
    print(np.mean(w))
    '''
    #print 'perfect_score:', perfect_score(data_y)
    print(Counter(data_y))

    #dtrain = xgb.DMatrix(data_x, label=data_y, weight=w)
    #dtest = xgb.DMatrix(test_x)
    #watchlist = [(dtrain, 'train')]
    bst = svm_FNC.fit(data_x, data_y)
    joblib.dump(bst, 'svm_cluster_train_takeout4.pkl')
    #bst = joblib.load('svm_cluster_train_tfidf.pkl')
    '''
    bst = xgb.train(params_xgb,
                    dtrain,
                    n_iters,
                    watchlist,
                    feval=eval_metric,
                    verbose_eval=10)
'''
    #pred_y = bst.predict(dtest) # output: label, not probabilities
    #pred_y = bst.predict(dtrain) # output: label, not probabilities
    pred_y = bst.predict(test_x)
    print(pred_y)
    '''pred_prob_y = pred_y.reshape((len(test_x), 4)) # predicted probabilities
    #pred_y = np.asarray((pred_prob_y))
    print('pred_y.shape:')
    print(pred_y.shape)'''
    predicted = [LABELS[int(a)] for a in pred_y]

    # save (id, predicted and probabilities) to csv, for model averaging
    #stances = pd.read_csv("test_stances_unlabeled_processed.csv") # same row order as predicted
    stances = pd.read_csv("test_stances_unlabeled.csv")

    df_output = pd.DataFrame()
    df_output['Headline'] = stances['Headline']
    df_output['Body ID'] = stances['Body ID']
    df_output['Stance'] = predicted
    '''
    df_output['prob_0'] = pred_prob_y[:, 0]
    df_output['prob_1'] = pred_prob_y[:, 1]
    df_output['prob_2'] = pred_prob_y[:, 2]
    df_output['prob_3'] = pred_prob_y[:, 3]
    '''
    #df_output.to_csv('submission.csv', index=False)
    df_output.to_csv('tree_pred_prob_cor2.csv', index=False)
    df_output[['Headline','Body ID','Stance']].to_csv('tree_pred_cor2.csv', index=False)

    print(df_output)
    print(Counter((df_output['Stance'])))

    #pred_train = bst.predict(dtrain).reshape(data_x.shape[0], 4)
    #pred_t = np.argmax(pred_train, axis=1)
    #predicted_t = [LABELS[int(a)] for a in pred_t]
    #print(Counter(predicted_t))

if __name__ == '__main__':
    #build_test_data()
    #cv()
    svm_FNC = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state= 2018)
    train()
