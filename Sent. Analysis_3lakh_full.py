#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import numpy as np
import pandas as pd
import keras
from tqdm import tqdm


# In[3]:


train = pd.read_csv(r"/home/amit/Documents/amazon_review_full_csv/train_3lakh_bin.csv")
train.head()


# In[4]:


#Make everything lowercase.
train['review'] = train['review'].str.lower()


# In[6]:


import re
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


# In[7]:


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


# In[ ]:


for i in range(len(train)):
    train['review'][i] = replace_contractions(train['review'][i])
    if(i%500 == 0):
        print("Step ", i/500)


# In[ ]:


#Remove all non-letter characters
train['review'] = train['review'].str.replace('[^a-zA-Z]',' ')


# In[7]:


#most frequent words
freq = pd.Series(' '.join(train['review']).split()).value_counts()[:80]
freq


# In[16]:


#removing most frequent words
freq = list(freq.index)
#print("removing...")
train['review'] = train['review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#print("removed.")
#train['review'].head(15)


# In[17]:


train['review'] = train['review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
train.head()


# In[42]:


#most rare words
freq = pd.Series(' '.join(train['review']).split()).value_counts()[-133817:]
freq 


# In[43]:


#removing rare words
freq = pd.Series(' '.join(train['review']).split()).value_counts()[-133817:]
freq = list(freq.index)
train['review'] = train['review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


# In[ ]:





# In[3]:


#saving a file after preprocessing
train.to_csv(r'/home/amit/Documents/amazon_review_full_csv/train_3lakh_bin_WR.csv', index=False) 


# In[15]:


train = pd.read_csv(r'/home/amit/Documents/amazon_review_full_csv/train_3lakh_bin_WR.csv') 
#train1.shape


# In[16]:


texts = train['review']


# In[17]:


import gensim.corpora as corpora
from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel


# In[18]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# In[19]:


stop_words = ["myself", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "him", "his", "himself",
 "she", "her", "hers", "herself", "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
 "which", "who", "whom", "this", "that", "these", "those", "are", "was", "were", "been", "being", "have", "has",
 "had", "having", "does", "did", "doing", "the", "and", "but", "because", "until", "while", "for", "with", "about",
 "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "down", "out",
 "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
 "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "than", "too",
 "very", "can", "will", "just", "don", "should", "now"]


# In[22]:


get_ipython().run_cell_magic('time', '', 'data_words = list(sent_to_words(texts))')


# In[23]:


for i in range(len(train)):
    data_words[i] = [word for word in data_words[i] if word not in stop_words]


# In[25]:


data_words


# In[ ]:





# In[24]:


id2word = corpora.Dictionary(data_words)


# In[10]:


corpus = [id2word.doc2bow(text) for text in data_words]


# In[11]:


print(corpus[:1])


# In[12]:


id2word[20]


# In[13]:


[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# In[ ]:


lda_train = gensim.models.ldamulticore.LdaMulticore(
                           corpus=corpus,
                           num_topics=100,
                           id2word=id2word,
                           chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = None,
                           per_word_topics=True)
lda_train.save('lda_train.model')


# In[ ]:


#lda_train.print_topics(20,num_words=15)[:10]


# In[ ]:


train_vecs = []
for i in range(len(train)):
    top_topics = lda_train.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(100)]
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
    lr = LogisticRegression(
#         class_weight= 'balanced',
#         solver='newton-cg',
#         fit_intercept=True
    ).fit(X_train_scale, y_train)

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


# In[ ]:


import os,datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard


# In[ ]:


y_bin = to_categorical(y_train)
y_bin_test = to_categorical(y_test)
y_bin_val = to_categorical(y_val)


# In[ ]:


hidden_nodes = 200
model = Sequential()
model.add(Dense(hidden_nodes, activation='sigmoid', input_dim=200))
model.add(Dense(hidden_nodes, activation='sigmoid',name = 'embedding'))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


import tensorflow as tf
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0, write_graph=False, write_images=True, embeddings_data= X_train_scale)


# In[ ]:


model.fit(X_train_scale, y_bin, 
                 validation_data=(X_val_scale, y_bin_val),epochs = 7, callbacks=[tensorboard])


# In[ ]:


score,acc = model.evaluate(X_test_scale, y_bin_test)
print("Test acc: %.3f" % (acc))

