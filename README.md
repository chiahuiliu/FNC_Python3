# claim_checking_Fall2018
This is the Fall 2018 repo for the evaluation metric subproject within the Claim Checking Group at UT Austin.

### Dev Environment
Please download Anaconda, and create a Python3 virtual evironment with the packages listed below. Here's a handy cheat sheet for creating and using Anaconda virtual environment.
https://conda.io/docs/_downloads/conda-cheatsheet.pdf


### For data, please download the data here
https://drive.google.com/drive/folders/1F2RVsVsOEOyq4xUG_taNcqccm5LHH27w?usp=sharing

### Prerequisites
- dill
- nltk
- pandas
- numpy
- ngram


### File explanations
`merge_csv.py`: merge all the test & training data from the fake news challenge dataset, and then merge it with the Emergent dataset.

`fnc_data.csv`: the merged data for "Fake News Challenge"
`merged_w_allCols.csv`: the merged dataset with all columns

*Use `merged_data_tain.csv` for training!!!
#### Columns in merged_data_tain.csv
- claimId: The unique sequence number for the claim
- claimHeadline: the textual content of the claim
- articleId: the unique sequence number for the article corresponding to the claim
- articleHeadline: the textual content of the article
(p.s one claim may have more than one article)
- claimTruthiness: the TRUE stance for the claim
- articleStance: the stance of the article
