#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_csv(r"/home/experiment/Amit/train_3lakh_bin_W_R.csv")


# In[3]:


texts = train['review']


# In[4]:


import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# In[5]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# In[6]:

data_words = list(sent_to_words(texts))


# In[7]:


stop_words = ["myself", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "him", "his", "himself",
 "she", "her", "hers", "herself", "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
 "which", "who", "whom", "this", "that", "these", "those", "are", "was", "were", "been", "being", "have", "has",
 "had", "having", "does", "did", "doing", "the", "and", "but", "because", "until", "while", "for", "with", "about",
 "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "down", "out",
 "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
 "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "than", "too",
 "very", "can", "will", "just", "don", "should", "now"]


# In[ ]:

#removing stopwords\n
for i in range(len(train)):    
    data_words[i] = [word for word in data_words[i] if word not in stop_words]


# In[ ]:

print("Stopwords removed \n")
# In[ ]:


id2word = corpora.Dictionary(data_words)


# In[ ]:


corpus = [id2word.doc2bow(text) for text in data_words]


# In[ ]:


print(corpus[:1])


# In[ ]:


id2word[20]


# In[ ]:


[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# In[ ]:


lda_train = gensim.models.ldamulticore.LdaMulticore(
                           corpus=corpus,
                           num_topics=200,
                           id2word=id2word,
                           chunksize=200,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = None,
                           per_word_topics=True)
lda_train.save('lda_train.model')
print("lda training done \n ")

# In[ ]:

train_vecs = []
for i in range(len(train)):
    top_topics = lda_train.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(200)]
    train_vecs.append(topic_vec)


# In[ ]:


#splitting dataset 
from sklearn.model_selection import train_test_split
train_rev, test_rev, train_rt, test_rt, = train_test_split(train_vecs,
                                                  train.rating,
                                                  test_size = .2,
                                                  random_state=12)


# In[ ]:


#X = np.array(train_vecs)
X = np.array(train_rev)
#y = np.array(train.rating)
y = np.array(train_rt)

X_test = np.array(test_rev)
y_test = np.array(test_rt)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score 

from sklearn.linear_model import LinearRegression
from sklearn import svm

kf = KFold(5, shuffle=True, random_state=42)
cv_lr_f1, cv_svm_f1, cv_lr_ac, cv_svm_ac, lr_test_ac, svm_test_ac = [], [], [], [], [], []



# In[ ]:


i = 0
for train_ind, val_ind in kf.split(X, y):
    # Assign CV IDX
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind]
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_val_scale = scaler.transform(X_val)
    X_test_scale = scaler.transform(X_test)

    # Logisitic Regression
    lr = LogisticRegression().fit(X_train_scale, y_train)

    y_pred = lr.predict(X_val_scale)
    test_pred = lr.predict(X_test_scale)
    cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))
    cv_lr_ac.append(accuracy_score(y_val, y_pred))
    lr_test_ac.append(accuracy_score(y_test, test_pred))
        
    #SVM
    model = svm.SVC(kernel='linear', C=1, gamma=1) 
    model.fit(X_train_scale, y_train)
    y_pred1 = model.predict(X_val_scale)
    test_pred1 = model.predict(X_test_scale)
    
    cv_svm_f1.append(f1_score(y_val, y_pred1, average='binary'))
    cv_svm_ac.append(accuracy_score(y_val, y_pred1))
    svm_test_ac.append(accuracy_score(y_test, test_pred1))
    
    i = i+1
    print("Step ", i)
print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'SVM Classifier f1: {np.mean(cv_svm_f1):.3f} +- {np.std(cv_svm_f1):.3f}')
print(f'Logistic regression Accuracy: {np.mean(cv_lr_ac):.3f} +- {np.std(cv_lr_ac):.3f}')
print(f'SVM Accuracy: {np.mean(cv_svm_ac):.3f} +- {np.std(cv_svm_ac):.3f}')
print(f'Logistic regression test Accuracy: {np.mean(lr_test_ac):.3f} +- {np.std(lr_test_ac):.3f}')
print(f'SVM test Accuracy: {np.mean(svm_test_ac):.3f} +- {np.std(svm_test_ac):.3f}')
