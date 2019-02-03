# Updated/Augmented Implementation of FNC Winning Solution

## Problem Statement
(Taken from http://www.fakenewschallenge.org)  
The goal of the Fake News Challenge Competition is to explore how machine learning and artificial intelligence technologies might be leveraged to combat the fake news problem. A helpful first step towards identifying fake news is to understand what other news organizations are saying about the topic. We believe automating this process, called Stance Detection, could serve as a useful building block in an AI-assisted fact-checking pipeline. 

## Dataset
The Fake News Challenge competition provided a dataset to participants. It was a subset of the Emergent dataset (http://www.aclweb.org/anthology/N16-1138), with an extra field -- article stance. We merged these two datasets (accessible here: https://tinyurl.com/yb5ldtf3) to generate an enhanced dataset containing article headlines, claim, article stance, and claim veracity.

## Feature Explanation
We refactored the FNC Winning Team's tree solution (available here in Python 2: https://github.com/Cisco-Talos/fnc-1) as well as added an simple SVM classifier as our two baseline models. Read more about their feature engineering in their repository README. Here's a basic explanation of each of the features used:

CountFeatureGenerator: Count occurrences of selected words in the data files  

SentimentFeatureGenerator: Using nltk's sentiment analysis tools, this file calculates the sentiments (compound, negative, neutral, positive) for each training example and returns the probability of each sentiment from the training data.  

TfidfFeatureGenerator: Calculates the tfidf score (https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for each claimHeadline and articleHeadline.   

SvdFeatureGenerator: Using the similarity score calculated from tfidf and SVD (Singular Value Decomposition) from the sklearn package, this file returns the SVD vector for claimHeadlines and articleHeadlines.  

Word2VecFeatureGenerator: This file returns the Word2Vec (https://en.wikipedia.org/wiki/Word2vec) representation from the claimHeadline and articleHeadline as well as the cosine similarity vectors between each articleHeadline and claimHeadline.  

## Implementation Guide + Instructions
### Dev Environment
Please download Anaconda, and create a Python 3 virtual evironment with the packages listed below. Here's a handy cheat sheet for creating and using Anaconda virtual environments:
https://conda.io/docs/_downloads/conda-cheatsheet.pdf

### Prerequisite Packages
Type `pip install -r rquirement.txt` to install all the required packages.
- dill
- nltk
- pandas
- numpy
- ngram
- Please download GoogleNews-vectors-negative300.bin in order to be able to successfully run the code.

### Feature Generator Code Flow
<img src="https://github.com/chiahuiliu/claim_checking_Fall2018/blob/feature_generation/claim_check_codeFlow.png"/>

### File explanations
`merge_csv.py`: merge all the test & training data from the fake news challenge dataset, and then merge it with the Emergent dataset.

`fnc_data.csv`: the merged data for "Fake News Challenge"

`merged_w_allCols.csv`: the merged dataset with all columns

`CountFeatureGenerator.py`: Count occurrences of selected words in the data files

`SentimentFeatureGenerator.py`: Using nltk's sentiment analysis tools, this file calculates the sentiments (compound, negative, neutral, positive) for each training example and returns the probability of each sentiment from the training data.

`TfidfFeatureGenerator.py`: Calculates the tfidf score (https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for each claimHeadline and articleHeadline.
Be sure to uncomment line 134 when running generateFeatures.py, and uncomment line 136 when running svm_2.py or other models.

`SvdFeatureGenerator.py`: Using the similarity score calculated from tfidf and SVD (Singular Value Decomposition) from the sklearn package, this file returns the SVD vector for claimHeadlines and articleHeadlines.

`Word2VecFeatureGenerator.py`: This file returns the Word2Vec (https://en.wikipedia.org/wiki/Word2vec) representation from the claimHeadline and articleHeadline as well as the cosine similarity vectors between each articleHeadline and claimHeadline.

`generateFeatures.py`: Driver for feature generation. Calculates the unigram, bigram, and trigram features, and also generates the features from files listed above.

*Use `merged_data_train.csv` for training!!!
#### Columns in merged_data_train.csv
- claimId: The unique sequence number for the claim
- claimHeadline: the textual content of the claim
- articleId: the unique sequence number for the article corresponding to the claim
- articleHeadline: the textual content of the article
(p.s one claim may have more than one article)
- claimTruthiness: the TRUE stance for the claim
- articleStance: the stance of the article

## Models

### Gradient Boosted Decision Tree Model
For xgboost documentation, refer to this: https://xgboost.readthedocs.io/en/latest/  
For an explanation of the GBDT model, read this: https://xgboost.readthedocs.io/en/latest/tutorials/model.html  

The main file for this model is `xgb_train_cvBodyId.py`. The parameters are set from line 23-34, we've modified the parameters in the subsequent 10 lines. Refer to the file for commented instructions.  

### SVM Model
For SVM sklearn documentation/model explanation, refer to this: https://scikit-learn.org/stable/modules/svm.html.  

The main file for this model is `svm_2.py`. The parameters are set in the main function. Again, refer to the file for commented instructions.  

### Evaluation Metric

The original evaluation metric for this task was a multi-level weighted score. Upon further investigation, we found inconsistencies with this metric. Read more about these inconsistencies in this paper (https://arxiv.org/pdf/1806.05180.pdf). 
We performed sensitivity analysis for multiple models, including the XGBoost, SVM, and logistic regression models, and found that there was an imbalance in the F1 score. The paper highlights other evaluation metrics as well.  

The main file for model evaluation is `score.py`. Upon running it, it will print an F1 score as well as a confusion matrix.
