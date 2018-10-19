# claim_checking_Fall2018
This is the Fall 2018 repo for the evaluation metric subproject within the Claim Checking Group at UT Austin.

### Problem Statement

We're trying to do several things:
1) Merge the Emergent dataset with the Fake News Challenge dataset (accessible through the Google doc) to generate an enhanced dataset containing article headlines, claim headline, article stance, and the veracity of the claim headline.
2) The original evaluation metric for this task was a multi-level weighted score. After iterating on this eval metric, we decided to performing sensitivity analysis for multiple models, including xgboost, SVM (Support Vector Machine), and logistic regression. We found that there was an imbalance in the F1 score, so this semester, we'd like to further develop an evaluation metric to assess the robustness of a given model. 

### Dev Environment
Please download Anaconda, and create a Python 3 virtual evironment with the packages listed below. Here's a handy cheat sheet for creating and using Anaconda virtual environments:
https://conda.io/docs/_downloads/conda-cheatsheet.pdf

### Prerequisite Packages
- dill
- nltk
- pandas
- numpy
- ngram

### For data, please download the data here
https://drive.google.com/drive/folders/1F2RVsVsOEOyq4xUG_taNcqccm5LHH27w?usp=sharing

### Code Flow Image
<img src="https://github.com/chiahuiliu/claim_checking_Fall2018/blob/feature_generation/claim_check_codeFlow.png"/>

### File explanations
`merge_csv.py`: merge all the test & training data from the fake news challenge dataset, and then merge it with the Emergent dataset.

`fnc_data.csv`: the merged data for "Fake News Challenge"

`merged_w_allCols.csv`: the merged dataset with all columns

`CountFeatureGenerator.py`: Count occurrences of selected words in the data files

`SentimentFeatureGenerator.py`: Using nltk's sentiment analysis tools, this file calculates the sentiments (compound, negative, neutral, positive) for each training example and returns the probability of each sentiment from the training data.

`TfidfFeatureGenerator.py`: Calculates the tfidf score (https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for each claimHeadline and articleHeadline.

`SvdFeatureGenerator.py`: Using the similarity score calculated from tfidf and SVD (Singular Value Decomposition) from the sklearn package, this file returns the SVD vector for claimHeadlines and articleHeadlines.

`Word2VecFeatureGenerator.py`: This file returns the Word2Vec (https://en.wikipedia.org/wiki/Word2vec) representation from the claimHeadline and articleHeadline as well as the cosine similarity vectors between each articleHeadline and claimHeadline.

`generateFeatures.py`: Driver for feature generation. Calculates the unigram, bigram, and trigram features, and also generates the features from files listed above.

*Use `merged_data_tain.csv` for training!!!
#### Columns in merged_data_tain.csv
- claimId: The unique sequence number for the claim
- claimHeadline: the textual content of the claim
- articleId: the unique sequence number for the article corresponding to the claim
- articleHeadline: the textual content of the article
(p.s one claim may have more than one article)
- claimTruthiness: the TRUE stance for the claim
- articleStance: the stance of the article
