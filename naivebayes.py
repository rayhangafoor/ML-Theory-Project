#!/usr/bin/env python
# coding: utf-8

# In[2]:


import io
import glob
import os
e_mails, labels = [], []
#load the spam e-mail files:
file_path = 'enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with io.open(filename, 'r', encoding = "ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(1)


# In[3]:


file_path = 'enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with io.open(filename, 'r', encoding = "ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(0)


# In[4]:


# print len(e_mails)
# print len(labels)


# #### Preprocess and Clean the raw text data steps:
#     Number and punctuation removal
#     Human name removal (optional)
#     Stop words removal
#     Lemmatization

# In[5]:


from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(wrds):
        return wrds.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()


# In[6]:


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_docs


# In[7]:


cleaned_e_mails = clean_text(e_mails)
# cleaned_e_mails[0]


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=500)
term_docs = cv.fit_transform(cleaned_e_mails)
# print(term_docs [0])


# In[9]:


feature_names = cv.get_feature_names()
feature_mapping = cv.vocabulary_


# In[10]:


def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index


# In[11]:


label_index = get_label_index(labels)


# In[12]:


#calculate the prior
def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.iteritems()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior


# In[13]:


prior = get_prior(label_index)
# prior = {0: 0.7099767981438515, 1: 0.2900232018561485}


# In[15]:


import numpy as np
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in label_index.iteritems():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood


# In[16]:


smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)
len(likelihood[0])


# In[29]:


def get_posterior(term_document_matrix,prior,likelihood):	
  num_docs=term_document_matrix.shape[0]
  posteriors=[]
  for i in range(num_docs):
      posterior={key: np.log(prior_label) for key, prior_label in prior.items()}
      for label,likelihood_label in likelihood.items():
          term_document_vector=term_document_matrix.getrow(i)
          counts=term_document_vector.data
          indices=term_document_vector.indices
          for count, index in zip(counts, indices):
              posterior[label]+=np.log(likelihood_label[index])*count
      min_log_posterior = min(posterior.values())
      for label in posterior:
          try:
              posterior[label]=np.exp(posterior[label]-min_log_posterior)
          except:
              posterior[label] = float('inf')
      sum_posterior=sum(posterior.values())
      for label in posterior:
          if posterior[label] == float('inf'):
              posterior[label] = 1.0
          else:
              posterior[label]/=sum_posterior
      posteriors.append(posterior.copy())
  return posteriors


# In[30]:


# e_mails_test = [
#     '''Subject: flat screens
#     hello ,

#     please call or contact regarding the other flat screens...

#     requested .
#     trisha tlapek - eb 3132 b
#     michael sergeev - eb 3132 a
#     also the sun blocker that was taken away from eb 3131 a .
#     trisha should two monitors also michael .
#     thanks
#     kevin moore''',
#     '''Subject: having problems in bed ? we can help !
#     cialis allows men to enjoy a fully normal sex life without
#     having to plan the sexual act .
#     if we let things terrify us, life will not be worth living
#     brevity is the soul of lingerie .
#     suspicion always haunts the guilty mind .''',
# ]


# # In[31]:


# cleaned_test = clean_text(e_mails_test)
# term_docs_test = cv.transform(cleaned_test)
# posterior = get_posterior(term_docs_test, prior, likelihood)
# print(posterior)


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_e_mails, labels, test_size=0.33, random_state=42)


# In[33]:


# print len(X_train), len(Y_train)
# print len(X_test), len(Y_test)


# In[34]:


term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing)

term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test, prior, likelihood)


# In[35]:


correct = 0.0
for pred, actual in zip(posterior, Y_test):
    if actual == 1:
        if pred[1] >= 0.5:
            correct += 1
    elif pred[0] > 0.5:
        correct += 1
print('The accuracy on {0} testing samples is:{1:.1f}%'.format(len(Y_test), correct/len(Y_test)*100))


# In[25]:


# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB(alpha=1.0, fit_prior=True)
# clf.fit(term_docs_train, Y_train)
# prediction_prob = clf.predict_proba(term_docs_test)
# prediction = clf.predict(term_docs_test)
# accuracy = clf.score(term_docs_test, Y_test)
# print accuracy


# # In[26]:


# from sklearn.metrics import precision_score, recall_score, f1_score
# precision_score(Y_test, prediction, pos_label=1)
# recall_score(Y_test, prediction, pos_label=1)
# f1_score(Y_test, prediction, pos_label=1)


# # In[ ]:


# pos_prob = prediction_prob[:, 1]
# thresholds = np.arange(0.0, 1.2, 0.1)
# true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
# for pred, y in zip(pos_prob, Y_test):
#     for i, threshold in enumerate(thresholds):
#         if pred >= threshold:
#         # if truth and prediction are both 1
#             if y == 1:
#                 true_pos[i] += 1
#         # if truth is 0 while prediction is 1
#             else:
#                 false_pos[i] += 1
#         else:
#             break


# # In[ ]:


# true_pos_rate = [tp / 516.0 for tp in true_pos]
# false_pos_rate = [fp / 1191.0 for fp in false_pos]


# # In[ ]:


# import matplotlib.pyplot as plt
# plt.figure()
# lw = 2
# plt.plot(false_pos_rate, true_pos_rate, color='darkorange',
# lw=lw)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()


# # In[ ]:




