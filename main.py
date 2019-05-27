
# coding: utf-8

# In[1]:


from preprocessor import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[2]:


def build_feats(projects, category):
    
    # Find and print most common unigrams and bigrams in category
    grams = grams_by_category(projects, category, do_print=False)

    # Map grams to unique index for easy vectorization
    grams_to_idx = map_gram_to_idx(grams)

    # Map unique index to gram to quickly convert vectorization to txt
    idx_to_grams = [0] * len(grams_to_idx)

    for gram, idx in grams_to_idx.items():
        idx_to_grams[idx] = gram
        
    # Build feats + labels for model training
    feats = []
    labels = []

    for project in projects:
        if project['category'][0] == category or category == 'all':
            encoding = vectorize(project, grams_to_idx)

            # Label represents amt pledged
            label = project['pledged']

            feats.append(encoding)
            labels.append(label)
            
    return idx_to_grams, feats, labels


# In[3]:


def create_model(projects, category, validate=False):
            
    idx_to_grams, feats, labels = build_feats(projects, category)
            
    # 90-10 split feats and labels; 90% training data and 10% test data
    feats_train = feats[:int(len(feats) * .9)]
    feats_test  = feats[int(len(feats) * .9):]

    labels_train = labels[:int(len(labels) * .9)]
    labels_test  = labels[int(len(labels) * .9):]
    
    model = linear_model.Ridge(alpha=1000)     # Initialize model
    model.fit(feats_train, labels_train)       # Train model
    
    # If validate=True, then validate model using 10% of data
    if validate:
        predictions = model.predict(feats_test)
        
        MSE = mean_squared_error(predictions, labels_test)
        print("MSE:", MSE)
        
    word_corrs = sorted(zip(idx_to_grams, model.coef_), key=lambda t: -t[1])
        
    return model, word_corrs


# In[4]:


projects = read_data()


# In[5]:


all_categories = []

# Get list of all possible categories
for project in projects:
    for category in project['category']:
        all_categories.append(category)
        
all_categories = nltk.FreqDist(all_categories)

# Get top-10 categories
top_10_categories = [category[0] for category in all_categories.most_common(10)]


# In[21]:


grams = {}
coefs = []

for category in top_10_categories:
    temp = {}
    
    LR, corrs = create_model(projects,category)
    
    temp['grams'] = [t[0] for t in corrs]
    temp['monetary_impact'] = [t[1] for t in corrs]
    
    coefs.append([category, LR.intercept_,LR.coef_[-1]])
    
    grams[category] = pd.DataFrame(temp)


# In[39]:


grams_df = pd.concat(grams, axis=1, keys=top_10_categories)
coefs_df = pd.DataFrame.from_records(coefs, columns=['category', 'intercept', 'goal_v_raised'], index='category')


# In[43]:


grams_df.style


# In[40]:


coefs_df.style


# In[42]:


# How to access intercept or goal_v_raised
print(coefs_df.loc['art']['intercept'])

# How to access grams
print(grams_df['art']['grams'][0])


# 
# plt.figure(figsize=(20,10))
# plt.bar(df_top['Gram'], df_top['Monetary Impact']) 
# plt.xlabel("Gram") 
# plt.ylabel("Monetary Impact") 
# plt.show() 

# 
# plt.figure(figsize=(20,10))
# plt.bar(df_bottom['Gram'], df_bottom['Monetary Impact']) 
# plt.xlabel("Gram") 
# plt.ylabel("Monetary Impact")
# plt.figure(figsize=(20,10))
# plt.show() 
# Actual project finder. If you're confused by a word, check this out
#word = 'camera'
#category = 'games'
#for i in range(len(projects)):
    #if word in projects[i]['text_feats'] and category in projects[i]['category']:
        #print(projects[i]['text'])
        #print(projects[i]['pledged'], "$ / ", projects[i]['goal'], "$")
        #print()