# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:34:40 2017

@author: baiintern1
"""
import Estimator
from sklearn import model_selection,svm
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def split_data(filename,n):
    """Gets the data from the filename and splits into training data and testing data
    
        Args:
            filename(string): the file where the data is located in csv format
            n(int or float): int for the test set size or float for the proportion of the data in the test set
            
        Returns:
            (tuple of pandas dataframes): has the training data in position 0 and the testing data in position 1"""
    data_pd = pd.read_csv(filename)
    training_pd, testing_pd = model_selection.train_test_split(data_pd,test_size = n, random_state=865)#865,7652,322
    return training_pd,testing_pd

def p_values(tbl1,tbl2):
    """Checks the the data is split in an even way
    
        Args:
            tbl1(pandas dataframe): either the training or testing set
            tbl2(pandas dataframe): the other set to compare it to"""
    def ttest(tbl1,tbl2,col):
        return ttest_ind(tbl1[col],tbl2[col])[1]
    
    def chisquared(tbl1,tbl2,col,length, index):
        obs1 = [0]*length
        obs2 = [0]*length
        col1, col2 = tbl1[col],tbl2[col]
        for i in col1:
            obs1[i+index]+=1
        for i in col2:
            obs2[i+index]+=1
        return chi2_contingency([obs1, obs2])[1]

    def checkpvalues(tbl1,tbl2):
        good = True
        for i in ["PTEDUCAT","BLage","MMSE v1","CDR_SOB_v1","ADASMOD v1","AVLT-TL v1","AVLT-LTM v1"]:
            val = ttest(tbl1,tbl2,i)
            print(val)
            if val<.05:
                good = False
        val1 = chisquared(tbl1,tbl2, "PTGENDER", 2, -1)
        print(val1)
        val2 = chisquared(tbl1,tbl2, "APOE Status", 3, 0)
        print(val2)
        val3 = chisquared(tbl1,tbl2, "AV45 v1 Pos118", 2, 0)
        print(val3)
        if val1<.05 or val2<.05 or val3<.05:
            good = False
        return good
    
    print(checkpvalues(tbl1,tbl2))
    


def classifier(model,train,test,scoring,k,n,col):
    """creates the model and trains it and then tests it
    
        Args: 
            model(string): the model to use to classify-possible choices are 
                svm for svm,mlp for mlp-classifier, forests for randomforestclassifier, and knn for knearestneighbors
            train(pandas dataframe): the data to train the model with
            test(pandas dataframe): the data to test the model with
            scoring(string): how to prioritize what to optimize 
                for more options, check http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            k(int): the amount of features to select
            n(int): the number of dimensions to reduce to
            col(string): the column you are trying to predict
            
        Returns:
            (estimator): returns the model if its needed for future puposes
            """
    if model=="svm":
        classify = svm.SVC()
        parameters = {'kernel':['rbf'],'C':[2**i for i in range(0,20)], 
                  'gamma':[2**i for i in range(-20,0)]}
    elif model=="mlp":
        classify = MLPClassifier(random_state=19654)
        parameters = {'hidden_layer_sizes':[60,70,80,90,[80,30],100],'alpha':[2**i for i in range(-13,14)]}
    elif model=="forest":
        classify = RandomForestClassifier(random_state=154)
        parameters = {'n_estimators':[30,35,40,42,45]}
    elif model=="knn":
        classify = KNeighborsClassifier()
        parameters = {'n_neighbors':range(1,15), 'weights': ['uniform', 'distance']}
    else:
        print("no model recognized")
        return
    GS = GridSearchCV(classify,parameters,verbose=2,scoring = scoring)
    est = Estimator.estimator(col,GS,k,True,n)
    est.fit(train)
    print(est.estimator.best_estimator_)
    est.check_model(test)
    est.plot()
    return est
    
        
def combine(preds,k,test):
    """combines several MLclassifiers in order to make one classifier that is more consistent
        Args:
            preds(list of lists): has the list of predicted classes from other classifiers
            k(int): the number of positive agreeing classes to make the combine class give positive
            test(pandas dataframe): the testing set that classifier is trying to classify
        
        Returns:
            (double): returns the f_score of the the combined predicted data"""
    pred = []
    for i in range(len(preds[0])):
        total = 0
        for j in preds:
            total+=j[i]
        if total>=k:
            pred.append(1)
        else:
            pred.append(0)
    
    index = test.shape[1]-1
    y = test.values[:,index]
    for i in range(len(y)):
        print(pred[i],y[i])
    print(1-sum(abs(pred-y))/len(pred))
    f = f1_score(y,pred)
    print(f)
    return f
    
    
def main():
    train,test = split_data("/Users/baiintern1/Documents/Freesurfer_volume/brainscan_data/normal.csv",50)
    p_values(train,test)
    svm1 = classifier("svm",train,test,"f1",23,6,"AV45 v1 Pos118").pred
    mlp1 = classifier("mlp",train,test,"f1",24,9,"AV45 v1 Pos118").pred
    #ft1 = classifier("forest",train,test,"f1",23,11,"AV45 v1 Pos118").pred
    #knn1 = classifier("knn",train,test,"f1",12,2,"AV45 v1 Pos118").pred
    combine([svm1,mlp1],2,test)
    


