import pandas as pd

#######################
# For Fake news challenge
# The key for merging is "Body_id"
# the output file is saved as fnc_data.csv
#######################
def read_csvs():
    # for Fake news challenge data
    # they separated data into the training and testing data set
    # for both train/test data, they are both consisted of _stances and _bodies files
    df_challenge_train_stance = pd.read_csv('data/train_stances.csv')
    df_challenge_train_body = pd.read_csv('data/train_bodies.csv', encoding='unicode_escape')
    df_challenge_test_stance = pd.read_csv('data/test_stances.csv')
    df_challenge_test_body = pd.read_csv('data/test_bodies.csv', encoding='unicode_escape')
    df_fnc_stance = df_challenge_train_stance.append(df_challenge_test_stance, ignore_index=True)
    df_fnc_bodies = df_challenge_train_body.append(df_challenge_test_body, ignore_index=True)
    return df_fnc_stance, df_fnc_bodies

def merge_fnc_article_body(article_df, body_df):
    merged_df = article_df.merge(body_df, left_on='Body ID', right_on='Body ID', how='left')
    return merged_df

df_fnc_stance,  df_fnc_bodies= read_csvs()
df_fnc = merge_fnc_article_body(df_fnc_stance, df_fnc_bodies)

df_fnc.to_csv('./data/fnc_data.csv', index=False)
