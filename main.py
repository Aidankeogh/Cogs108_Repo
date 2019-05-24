#!/usr/bin/env python
# coding: utf-8

# In[11]:


from preprocessor import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model


# In[12]:


projects = read_data()
category = "games"


# In[13]:


# Choose and print out a random sample from the set
i = random.randint(0, len(projects)) 

print(projects[i]['text'])
print()
print(projects[i]['text_feats'])
print()
print(projects[i]['category'])
print()
print(projects[i]['pledged'], "$ / ", projects[i]['goal'], "$")


# In[14]:


# Find and print most common uni-, bi-, and trigrams in category
grams = grams_by_category(projects, category, do_print=False)

# Map grams to unique index for easy vectorization
grams_to_idx = map_gram_to_idx(grams)

# Map unique index to gram to quickly convert vectorization to txt
idx_to_grams = [0] * len(grams_to_idx)

for gram, idx in grams_to_idx.items():
    idx_to_grams[idx] = gram


# In[15]:


# Build feats + labels for model training
feats = []
labels = []

for project in projects:
    
    if project['category'][0] == category:
        # Project encoding indicates which of the uni-, bi-, and 
        # trigrams in 'text_feats' are in the n-most common grams
        # for the category
        encoding = vectorize(project['text_feats'], grams_to_idx)
        
        # Label represents amt pledged
        label = project['pledged']
        
        feats.append(encoding)
        labels.append(label)


# In[16]:


# 90-10 split feats and labels; 90% training data and 10% test data
feats_train = feats[:int(len(feats) * .9)]
feats_test  = feats[int(len(feats) * .9):]

labels_train = labels[:int(len(labels) * .9)]
labels_test  = labels[int(len(labels) * .9):]


# In[17]:


# Train model
LR = linear_model.Ridge(alpha=1000)
LR.fit(feats_train, labels_train)


# In[18]:


print( "Expected pledge amt. if given NO project txt: %.2f" % LR.intercept_)
print() 


# In[19]:


zipped = sorted(zip(idx_to_grams, LR.coef_), key=lambda t: -t[1])

df = pd.DataFrame(zipped, columns=["Gram", "Monetary Impact"])
df.style

