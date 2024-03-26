'''
This code is based on an assignment from the University of Michigan's Coursera course titled Applied Machine Learning in Python. I have expanded on the assignment a bit, adding different kinds of classifiers, ROC curves, performance assessments, etc.

Skills demonstrated: classification, scikit-learn, TensorFlow, model evaluation, general coding

This code trains one or more classifiers to assess how engaging videos from a dataset are based on seven features defined below. A video is classified as "engaging" if the median percentage of the video watched across all viewers was at least 30%.

Dataset: engagement_data.csv:

Features:

    1. title_word_count - the number of words in the title of the video.

    2. document_entropy - a score indicating how varied the topics are covered in the video, based on the transcript. Videos with smaller entropy scores will tend to be more cohesive and more focused on a single topic.

    3. freshness - The number of days elapsed between 01/01/1970 and the lecture published date. Videos that are more recent will have higher freshness values.

    4. easiness - A text difficulty measure applied to the transcript. A lower score indicates more complex language used by the presenter.

    5. fraction_stopword_presence - A stopword is a very common word like 'the' or 'and'. This feature computes the fraction of all words that are stopwords in the video lecture transcript.

    6. speaker_speed - The average speaking rate in words per minute of the presenter in the video.

    7. silent_period_rate - The fraction of time in the lecture video that is silence (no speaking).

Target variable:

    1. engagement - Target label for training. True if learners watched a substantial portion of the video (see description), or False otherwise.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from engage_utils import *

#Prepare data
X_train, X_test, y_train, y_test = data_prep(pd.read_csv('engagement_data.csv'))

#Train and run classifiers
rec_svc, svc = engagement_model_SVM(X_train, X_test, y_train, y_test)

rec_lr, lr, cols, coefs = engagement_model_logReg(X_train, X_test, y_train, y_test)

rec_NN, NN = engagement_model_NN(X_train, X_test, y_train, y_test)

#Plot the size of each coefficient from the logistic regression classifier. The bigger the magnitude, the more important the coefficient (and corresponding feature)
if (coefs is not None):
    feature_importance = pd.DataFrame({'Feature': cols, 'Importance': np.abs(coefs)}).sort_values('Importance')
    feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
    
    
#Calculate and plot ROC curves for each classifier
fpr_svc, tpr_svc, opt_svc = roc_info(y_test,rec_svc)
fpr_lr, tpr_lr, opt_lr = roc_info(y_test,rec_lr)
fpr_nn, tpr_nn, opt_nn = roc_info(y_test,rec_NN)

roc_plotter(fpr_svc, tpr_svc, opt_svc,'SVC')
roc_plotter(fpr_lr, tpr_lr, opt_lr, 'Logistic regression')
roc_plotter(fpr_nn, tpr_nn, opt_nn, 'Neural Network')


#Compute and return the ROC_AUC and accuracy scores of each classifier
svc_auc = roc_auc_score(y_test, rec_svc)
svc_acc = accuracy_score(y_test, (rec_svc > opt_svc[2]))
lr_auc = roc_auc_score(y_test, rec_lr)
lr_acc = accuracy_score(y_test, (rec_lr >  opt_lr[2]))
nn_auc = roc_auc_score(y_test,rec_NN)
nn_acc = accuracy_score(y_test,(rec_NN > opt_nn[2]))

print('We can compare the AUC and accuracy scores of each model:')

print(f'SVC AUC score: {svc_auc:.3f}')
print(f'SVC accuracy score at optimal probability threshold: {svc_acc:.3f}')

print(f'LogisticRegression AUC score: {lr_auc:.3f}')
print(f'LogisticRegression accuracy score at optimal probability threshold: {lr_acc:.3f}')

print(f'Neural network AUC score: {nn_auc:.3f}')
print(f'Neural network accuracy score at optimal probability threshold: {nn_acc:.3f}')
