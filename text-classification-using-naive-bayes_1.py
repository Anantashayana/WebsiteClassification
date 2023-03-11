#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import chain

import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd

import seaborn as sns

from sklearn import feature_extraction
from sklearn.metrics import confusion_matrix
from sklearn import model_selection as ms
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import string


# <h3>Read in data, add header row and display first 5 rows:</h3>

# In[2]:


text_df = pd.read_csv('./website_classification.csv')
text_df.head(5)


# <h3>We are deleting two unnecessary columns from the dataframe: <I>'Unnamed: 0'</I> and <I>'website_url'</I></h3>

# In[3]:


del text_df['Unnamed: 0'], text_df['website_url']
text_df.head(5)


# <h4>Unique categories:</h4>

# In[4]:


categories = list(text_df['Category'].unique())
categories


# In[5]:


text_df.shape


# <h4>We have 1408 rows and 4 columns</h4>

# In[6]:


cleaned_website_text = list(text_df['cleaned_website_text'].unique())
len(cleaned_website_text)


# <h4>There are 1375 unique rows</h4>

# <h3>Number of rows by category</h3>

# In[7]:


text_df['Category'].value_counts()


# In[8]:


text_df.drop(text_df[text_df['Category']=='Forums'].index ,inplace = True)
text_df.drop(text_df[text_df['Category']=='Adult'].index ,inplace = True)


# <h3>Number of <B><I>unique</I></B> rows by category</h3>

# In[9]:


cat_unique_val = {}

for category in categories:
    mask = text_df['Category'] == category
    list_length = len(list(text_df[mask]['cleaned_website_text'].unique()))
    cat_unique_val.update({category: list_length})

    cat_unique_val


# <h3>Sort dictionary in descending order</h3>

# In[10]:


cat_unique_val_sorted = {}
sorted_keys = sorted(cat_unique_val, key=cat_unique_val.get)

for w in list(reversed(sorted_keys)):
    cat_unique_val_sorted[w] = cat_unique_val[w]
    
# cat_unique_val_sorted
cat_unique_val_sorted


# <h3>Dropping duplicate rows from each category</h3>

# In[11]:


text_df = text_df.drop_duplicates()


# In[12]:


text_df.shape


# <h3>Text cleaning</h3>

# <p>Removing stopwords and punctuation from <B><I>"cleaned_website_text" column</I></B></p>

# In[13]:


stop_words = set(stopwords.words('english'))


# In[14]:


regular_punct = list(string.punctuation)


# <h3>According to title text in our dataframe column is cleared, but in any case we will aplly function to remove posssible stopwords and punctuation</h3>

# In[15]:


text_df['cleaned_website_text']


# In[16]:


def text_preprocessing(x):
    filtered_sentence = []
    word_tokens = word_tokenize(x)
    
    for w in word_tokens:
        if w not in chain(stop_words, regular_punct):
            # we make sure that all words are written in lowercase
            filtered_sentence.append(w.lower())
    
    # Converting a list of strings back to a string
    filtered_sentence = " ".join(filtered_sentence)
    return filtered_sentence


# In[17]:


text_df['cleaned_website_text'] = text_df['cleaned_website_text'].apply(text_preprocessing)


# In[18]:


#nltk.download('all')


# In[19]:


text_df['cleaned_website_text']


# <h3>Preprocessing the data</h3>

# <h4>One of the simplest
# methods of encoding text data is by word count: For each phrase, you count the
# number of occurrences of each word within it. In scikit-learn, this is easily done using
# CountVectorizer:</h4>

# In[20]:


text_df['cleaned_website_text'].values


# In[21]:


text_df['cleaned_website_text'].shape


# In[22]:


counts = feature_extraction.text.CountVectorizer()
X = counts.fit_transform(text_df['cleaned_website_text'].values)
X.shape


# In[23]:


temp = counts


# In[24]:


array = X.toarray()


# In[25]:


df = pd.DataFrame(data=array,columns = counts.get_feature_names())


# In[26]:


df


# In[27]:


print(X)


# In[28]:


print(X.shape)


# In[29]:


type(X)


# In[30]:


y = text_df['Category'].values


# In[31]:


y


# 
# <h4>Training a normal Bayes classifier</h4>

# In[32]:


X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


model_naive = naive_bayes.MultinomialNB()
model_naive.fit(X_train, y_train)


# In[34]:


model_naive.score(X_train, y_train)


# In[35]:


model_naive.score(X_test, y_test)


# In[36]:


print(y_test)


# <h4><B><I>We got 88% accuracy on the test set</I></B></h4>

# <h4>Confusion matrix</h4>

# In[37]:


X_test.shape


# In[38]:


y_test.shape


# In[39]:


type(X_test)


# In[40]:


confusion_matrix(y_test, model_naive.predict(X_test))


# In[41]:


mat = confusion_matrix(y_test, model_naive.predict(X_test))
plt.figure(figsize=(18,12))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categories,
            yticklabels=categories
           )
plt.xlabel('true label')
plt.ylabel('predicted label');


# # Testing Data

# In[42]:


data = pd.read_excel('./Websites-list-test.xlsx')
data.to_csv('./websites_test.csv',index=False)
data=pd.read_csv('./websites_test.csv',index_col=False)
data.head()


# In[43]:


df = data.sample(n=10)


# In[44]:


for x in df.index:
    print(df['SITES'][x])


# In[45]:


for x in df.index:
    df['SITES'][x] = 'http://'+ df['SITES'][x]


# In[46]:


for x in df.index:
    print(df['SITES'][x])


# In[47]:


# Web Scraping

from bs4 import BeautifulSoup
import requests
import os
# Request to website and download HTML contents


def check_meta(x):
    url=x
    try:
        req=requests.get(url)
        content=req.text
        st=""
        soup=BeautifulSoup(content)
        f = open("./webdata.txt", "w", encoding="utf-8")
        for data in soup.find_all({"meta":"content"}):
            sum1 = data.get_text()
            st+=sum1
            f.writelines(sum1)
        L=st.split()
        f.close()
        f=open("./webdata.txt", "w", encoding="utf-8")
        for i in range(len(L)):
            f.writelines(L[i]+" ")
        if os.path.getsize("./webdata.txt") == 0:
            headcall(url)
        f.close()
        f=open("./webdata.txt", "r", encoding="utf-8")
        s=""
        for x in f.readlines():
            s+=x
        return s
    except:
        print("website Not found")
        return 'No'

    

def headcall(x):
    url=x
    try:
        req=requests.get(url)
        content=req.text
        st=""
        soup=BeautifulSoup(content)
        f = open("./webdata.txt", "w", encoding="utf-8")
        for data in soup.find_all("head"):
            sum1 = data.get_text()
            st+=sum1
            f.writelines(sum1)
        L=st.split()
        f.close()
        f=open("./webdata.txt", "w", encoding="utf-8")
        for i in range(len(L)):
            f.writelines(L[i]+" ")
        f.close()
        f=open("./webdata.txt", "r", encoding="utf-8")
        s=""
        for x in f.readlines():
            s+=x
        return s
    except:
        print("website Not found")
        return 'No'


# In[48]:


def clean(s):
    regular_punct = list(string.punctuation)
    stop_words = set(stopwords.words('english'))
    s=text_preprocessing(s)
    return s


# In[49]:


def vect(sar):
    ip=np.zeros((1,58533))
    ip = pd.DataFrame(data=ip,columns = temp.get_feature_names())
    for x in sar:
        if x in  temp.get_feature_names():
            ip[x]+=1
    return ip


# In[51]:


#Testing each website


for x in df.index:
    print("Website:"+df["SITES"][x])
    s = check_meta(df["SITES"][x])
    if s == 'No':
        print("Cannot be searched ")
    else:
        s=clean(s)
        print(s)
        sar = s.split()
        p=vect(sar)
        output=model_naive.predict(p)
        print("Category:"+str(output))
        print("\n")


# In[ ]:


s=""
url=input("Enter a website:")
url='http://'+url
s = check_meta(url)
if s == 'No':
    print("Cannot be searched ")
else:
    s=clean(s)
    print(s)
    sar = s.split()
    p=vect(sar)
    output=model_naive.predict(p)
    print("Category:"+str(output))
    print("\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




