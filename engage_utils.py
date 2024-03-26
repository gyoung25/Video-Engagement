import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def data_prep(X):
    '''
    Prepare data to be used to fit and test standard classifiers.
    
    Argument
        X: pandas DataFrame containing video engagement data
    Returns
        X_train_std_df X_test_std_df: Standardized training and test feature DataFrames
        y_train, y_test: training and test target Series
    '''
    
    X.set_index('id', inplace = True)
    X.drop(['normalization_rate'], axis = 1, inplace=True)
    y = X.pop('engagement')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    #print(y_train)
    
    #The features size can vary by several orders of magnitude between features, so we'll normalize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    X_train_std_df = pd.DataFrame(X_train_std, columns = X_train.columns, index = X_train.index)
    X_test_std_df = pd.DataFrame(X_test_std, columns = X_test.columns, index = X_test.index)
    
    return X_train_std_df, X_test_std_df, y_train, y_test



def engagement_model_SVM(X_train, X_test, y_train, y_test, search = False):
    
    '''
    Predict video engagement from data using a SVC model
    
    Arguments
        X_train, X_test, y_train, y_test: output from data_prep()
        search: Boolean that determines whether a GridSearchCV will be performed. If search = False, then the classifier is
                parameterized with the optimal C and gamma values determined from a prior GridSearchCV. Setting search to False                   saves a substantial amount of time. Defaults to false.
    The function returns 
        rec: a Series with values y_prob, indexed by the video ID
        clf: the fitted SVC classifier

    '''
    from sklearn.svm import SVC
    if search == True:
        svc = SVC(random_state=0, probability=True)
        # NOTE: could also include other SVC kernels in the params dictionary, 
        # but this code is primarily for illustrative purposes and adding more parameters
        # adds much more time to the grid search
        params = {'C':[0.001, 0.01, 0.1, 1], 'gamma':[0.001, 0.01, 0.1, 1]} 
        clf = GridSearchCV(svc, params, scoring = 'roc_auc')
    else:
        clf = SVC(C = 0.1, gamma = 0.01, random_state=0, probability=True)

    clf.fit(X_train, y_train)
    #params = clf.best_params_
    y_prob = clf.predict_proba(X_test)[:,1]
    
    rec = pd.Series(y_prob, index=y_test.index)
    return rec, clf




def engagement_model_logReg(X_train, X_test, y_train, y_test, search = False):
    '''
    Predict video engagement from data using a logistic regression model
    
    Arguments
        X_train, X_test, y_train, y_test: output from data_prep()
        search: Boolean that determines whether a GridSearchCV will be performed. If search = False, then the classifier is
                parameterized with the optimal C and gamma values determined from a prior GridSearchCV. Setting search to False                   saves a substantial amount of time. Defaults to false.
    Returns 
        rec - a Series with values y_prob, indexed by the video ID
        clf - the fitted LogisticRegression classifier
        X_train.columns - the feature names (for plotting purposes)
        coefs - the coefficients from the fitted logistic regression model to assess feature importance

    '''
    from sklearn.linear_model import LogisticRegression as LR
    
    if search == True:
        lr = LR(random_state=0)
        params = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10]} 
        clf = GridSearchCV(lr, params, scoring = 'roc_auc')
        clf.fit(X_train_std, y_train)
        print(clf.best_params_)
        coefs = None
    else:
        clf = LR(C = 0.1, random_state=0)
        clf.fit(X_train, y_train)
        coefs = clf.coef_[0]
    
    y_prob = clf.predict_proba(X_test)[:,1]
    
    rec = pd.Series(y_prob, index = y_test.index)

    return rec, clf, X_train.columns, coefs



def engagement_model_NN(X_train, X_test, y_train, y_test, num_epochs = 10, batch_size = 64):
    '''
    Predict video engagement from data using a logistic regression model
    
    Arguments
        X_train, X_test, y_train, y_test: output from data_prep()
        
    Returns 
        rec - a Series with values y_prob, indexed by the video ID
        clf - the fitted NN classifier

    '''
    import tensorflow as tf
    from keras import Sequential
    from keras.losses import BinaryCrossentropy
    from keras.metrics import AUC
    from keras.layers import Dense
    from keras.optimizers import Adam
    
    tf.keras.utils.set_random_seed(0)
    
    df = pd.read_csv('engagement_data.csv')
    df.set_index('id', inplace = True)
    df.drop(['normalization_rate'], axis = 1, inplace=True)
    #print(df.head())
    X = df.copy().drop(['engagement'], axis = 1)
    y = df['engagement']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    model = Sequential(
    [
        Dense(16, activation="relu", name="layer1"),
        Dense(64, activation="relu", name="layer2"),
        Dense(64, activation="relu", name="layer3"),
        Dense(32, activation="relu", name="layer4"),
        Dense(1, activation="sigmoid", name="output"),
    ]
    )
    
    model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=BinaryCrossentropy(),
              metrics=[AUC()])
    model.fit(X_train_std, y_train, validation_data=(X_test_std,y_test), batch_size=batch_size, epochs=num_epochs)

    score=model.evaluate(X_test_std, y_test,verbose=0)
    print(score[1])
    
    y_prob = model.predict(X_test_std)
    

    rec = pd.Series(tf.reshape(y_prob,[len(y_prob)]), index = y_test.index)

    return rec, model



def roc_info(y_test, y_prob):
    '''
    Plot the ROC curve for each classifier above and return the probability threshold value that minimizes the Euclidean
    distance to fpr = 0, tpr = 1, thereby optimizing the fpr-tpr tradeoff.
    
    Arguments
        y_test: test target Series
        y_prob: output of any classifier giving the probability that a video will be engaging
    Returns
        fpr: x-coordinates of the ROC curve
        tpr: y-coordinates of the ROC curve
        (x0, y0, thresh): the (x,y) coordinate and corresponding probability threshold value that minimizes the Euclidean                                   distance to fpr = 0, tpr = 1, thereby optimizing the fpr-tpr tradeoff
    '''
    from sklearn.metrics import roc_curve
    from numpy import where, abs, argmin
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # determine probability threshold that minimizes the distance to (fpr,tpr)=(0,1), 
    # which is the optimal threshold in the sense of tpr-fpr tradeoff
    dist = fpr**2 + (tpr-1)**2
    ind = argmin(dist)
    x0, y0, thresh = fpr[ind], tpr[ind], thresholds[ind]
    
    return fpr, tpr, (x0, y0, thresh)

def roc_plotter(fpr, tpr, opt_thresh, clf_type):
    '''
    Plot the ROC curve for each classifier above.
    
    Arguments
        fpr: x-coordinates of the ROC curve returned by roc_info()
        tpr: y-coordinates of the ROC curve returned by roc_info()
        opt_thresh: 3-tuple of the form (x0, y0, thresh), where the (x0,y0) is the coordinate and thresh is the corresponding                         probability threshold value that minimizes the Euclidean distance to fpr = 0, tpr = 1, thereby optimizing the                     fpr-tpr tradeoff returned by roc_info()
        clf_type: String used to label the corresponding ROC curve on the generated plot
    '''
    x0, y0, thresh = opt_thresh
    plt.plot(fpr, tpr, linewidth = 2, label = clf_type)
    plt.plot(x0, y0, 'o', markersize = 10)#, label = f'Threshold {thresh:.3f}')
    plt.legend()
    plt.title('ROC curves', size = 14)
    plt.xlabel('False Positive Rate', size = 14)
    plt.ylabel('True Positive Rate', size = 14)
    return None