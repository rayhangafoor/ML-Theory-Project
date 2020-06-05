#!/usr/bin/env python
# coding: utf-8

import io
import glob
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


e_mails, labels = [], []

#load the spam e-mail files:
file_path = 'enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with io.open(filename, 'r', encoding = "ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(1)

#load the ham e-mail files:
file_path = 'enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with io.open(filename, 'r', encoding = "ISO-8859-1") as infile:
        e_mails.append(infile.read())
        labels.append(0)


from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(wrds):
        return wrds.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_docs

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

#calculate the prior
def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.iteritems()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior


def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in label_index.iteritems():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

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



# true_pos_rate = [tp / 516.0 for tp in true_pos]
# false_pos_rate = [fp / 1191.0 for fp in false_pos]


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


k = 10
k_fold = StratifiedKFold(n_splits=k)
cleaned_e_mails = clean_text(e_mails)
cleaned_e_mails_np = np.array(cleaned_e_mails)
labels_np = np.array(labels)



max_features_option = [2000, 4000, 8000]
smoothing_factor_option = [0.5, 1.0, 1.5, 2.0]
fit_prior_option = [True]
auc_record = {}



for train_indices, test_indices in k_fold.split(cleaned_e_mails, labels):

    X_train, X_test = cleaned_e_mails_np[train_indices], cleaned_e_mails_np[test_indices]
    Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]

    for max_features in max_features_option:
        if max_features not in auc_record:
            auc_record[max_features] = {}
            cv = CountVectorizer(stop_words="english", max_features=max_features)
            term_docs_train = cv.fit_transform(X_train)
            term_docs_test = cv.transform(X_test)
            label_index = get_label_index(Y_train)
            prior = get_prior(label_index)

            for smoothing in smoothing_factor_option:
                if smoothing not in auc_record[max_features]:
                    auc_record[max_features][smoothing] = {}
                for fit_prior in fit_prior_option:
                    likelihood = get_likelihood(term_docs_train, label_index, smoothing)
                    term_docs_test = cv.transform(X_test)
                    posterior = get_posterior(term_docs_test, prior, likelihood)

                    correct = 0.0
                    for pred, actual in zip(posterior, Y_test):
                        if actual == 1:
                            if pred[1] >= 0.5:
                                correct += 1
                        elif pred[0] > 0.5:
                            correct += 1
                    auc = (correct/len(Y_test)*100)
                    auc_record[max_features][smoothing][fit_prior] = auc + auc_record[max_features][smoothing].get(fit_prior, 0.0)



print('max features   smoothing   fit prior   auc'.format(max_features, smoothing, fit_prior, auc))
for max_features, max_feature_record in auc_record.iteritems():
    for smoothing, smoothing_record in max_feature_record.iteritems():
        for fit_prior, auc in smoothing_record.iteritems():
            print('  {0}         {1}         {2}       {3:.4f}'.format(max_features, smoothing, fit_prior, auc))

