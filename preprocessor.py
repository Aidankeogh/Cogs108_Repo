
# coding: utf-8

# In[1]:


import json
import glob
import nltk
import random


# In[2]:


# A constant for the top most useful uni-, bi-, and trigrams
most_useful = {"uni": 500, "bi": 250, "tri": 25}


# **read_data()**
# - **Func Desc:**<br>
#     This function reads in the entire Kickstarter dataset from json files in the "kickstarter_data" directory.
# - **Return:**<br>
#     An nx5 list of projects, where n represents the total number of projects. Note that there are 5 attributes of a single project: the category, text, pledged amount, goal amount, and text_features.

# In[3]:


def read_data():
    projects = []

    # Read in data
    json_files = glob.glob("kickstarter_data/data*")

    for json_file in json_files:
        projects += json.load(open(json_file, 'r'))

    return projects    


# **grams_by_project(*list text*)**
# - **Func Desc:**<br>
#     This function will find all the unigrams, bigrams, and trigrams in the given *text*.
# - **Return:**<br>
#     A dictionary containing all unigrams, bigrams, and trigrams, 
#     where the corresponding keys are "uni", "bi" and "tri"

# In[4]:


def grams_by_project(text):
    grams = {}
    
    all_words = []
    all_bigrams = []
    all_trigrams = []
    
    prev_prev = ''
    prev_word = '<SOS>' # Start of sentence

    for w in text:
        # Ignore empty strings and apostrophe+s ending
        if w == "'s" or w == '’s' or w == '':  
            continue

        all_words.append(w)
        all_bigrams.append(prev_word + " " + w)

        if prev_prev != '':
            all_trigrams.append(prev_prev + " " + prev_word + " " + w)

        prev_prev = prev_word
        prev_word = w
    
    grams["uni"] = all_words
    grams["bi"]  = all_bigrams
    grams["tri"] = all_trigrams
    
    return grams


# **grams_by_category(*string category*, **[optional]** *int n*, **[optional]** *boolean do_print*)**
# - **Func Desc:**<br>
#     This function will find the unigrams, bigrams, and trigrams in the given *category*. If *do_print* is set, then the *n* most common unigrams, bigrams, and trigrams will be displayed.
# - **Return:**<br>
#     A dictionary containing all unigrams, bigrams, and trigrams, 
#     where the corresponding keys are "uni", "bi" and "tri"

# In[5]:


def grams_by_category(projects, category, n=15, do_print=True):
    grams = {}
    
    all_words = []
    all_bigrams = []
    all_trigrams = []
    
    for project in projects:
        
        # Change this to check out a different sub-category, 
        # 'all' will check the entire thing
        if category != 'all' and category not in project['category']: 
            continue

        prev_prev = ''
        prev_word = '<SOS>' # Start of sentence
        
        proj_grams = grams_by_project(project['text_feats'])
            
        all_words += proj_grams["uni"]
        all_bigrams += proj_grams["bi"]
        all_trigrams += proj_grams["tri"]
        
    grams["uni"] = nltk.FreqDist(all_words)
    grams["bi"]  = nltk.FreqDist(all_bigrams)
    grams["tri"] = nltk.FreqDist(all_trigrams)
    
    if do_print:
        print("-- UNIGRAMS --")
        all_words = nltk.FreqDist(all_words)
        
        for word in all_words.most_common(n):
            print(word[0], "\t", word[1])

        print()
        print("-- BIGRAMS --")
        all_bigrams = nltk.FreqDist(all_bigrams)
        
        for bigram in all_bigrams.most_common(n):
            print(bigram[0], "\t", bigram[1])

        print()
        print("-- TRIGRAMS --")
        all_trigrams = nltk.FreqDist(all_trigrams)
        
        for trigram in all_trigrams.most_common(n):
            print(trigram[0], "\t", trigram[1])
    
    return grams


# **map_gram_to_idx(*dictionary grams*, **[optional]** num_uni, **[optional]** num_bi, **[optional]** num_tri)**
# - **Func Desc:**<br>
#     Given a dictionary of unigrams, bigrams, and trigrams, this function maps each gram to a unique index. We will later use this to vectorize the most unique uni-, bi-, and trigrams. Note that *num_uni* represents the "n" most common unigrams, and similarily for *num_bi* and *num_tri*.
# - **Return:**<br>
#     A dictionary containing all unigrams, bigrams, and trigrams mapped to a unique integer index.

# In[6]:


def map_gram_to_idx(grams_dict, num_uni=most_useful["uni"], 
                      num_bi=most_useful["bi"], 
                      num_tri=most_useful["tri"]):
    gram_to_idx = {}
    count = 0
    
    for word, _ in grams_dict["uni"].most_common(num_uni):
        gram_to_idx[word] = count
        count += 1

    for phrase, _ in grams_dict["bi"].most_common(num_bi):
        gram_to_idx[phrase] = count
        count += 1

    for phrase, _ in grams_dict["tri"].most_common(num_tri):
        gram_to_idx[phrase] = count
        count += 1
        
    return gram_to_idx


# **vectorize(*list text*, *dictionary gram_to_idx*)**
# - **Func Desc:**<br>
#     For each uni-, bi-, and trigram in *text*, this function will indicate whether each gram is present in *gram_to_idx* (1: present; 0: not present). Note that *gram_to_idx* represents a mapping of the n most common uni-, bi-, and trigrams of a particular project category.
# - **Return:**<br>
#     A list of 0s and 1s, where 0 indicates that the gram found at *gram_to_idx[i]* is not present in *text* and 1 means that the gram is present.

# In[7]:


def vectorize(text, gram_to_idx):
    feats = [0] * len(gram_to_idx)    
    proj_grams = grams_by_project(text)
        
    for _, grams in proj_grams.items():
        for g in grams:
            if g in gram_to_idx:
                feats[gram_to_idx[g]] = 1
                
    return feats

