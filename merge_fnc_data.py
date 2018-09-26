
# coding: utf-8

# In[1]:


import pandas as pd


# # Merge Headline and Body text - Fake news challenge

# In[3]:


# for Fake news challenge data
df_challenge_train = pd.read_csv('data/train_stances.csv', encoding='unicode_escape')


# In[4]:


df_challenge_train.head()


# In[6]:


df_challenge_train['Body ID'].value_counts()


# In[5]:


df_challenge_train_bodies = pd.read_csv('data/train_bodies.csv', encoding='unicode_escape')


# In[7]:


df_challenge_train_bodies.head()


# In[8]:


df_challenge_train_bodies['Body ID'].value_counts()


# In[9]:


# merge
df_challenge_train = df_challenge_train.merge(df_challenge_train_bodies, left_on='Body ID', right_on='Body ID', how='inner')


# In[12]:


df_challenge_train.sample(10)


# In[14]:


df_challenge_train.to_csv("data/merge_fnc.csv", index=False)

