# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:31:57 2017

@author: Noah Krakoff
"""

from sklearn import preprocessing,model_selection
from sklearn.metrics import f1_score
import pandas as pd
import Feature_selection
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class estimator():
    """This estimator is actually a generalizaton for machine learning classifier, but
     there are a few restrictions:
         The column data in each table must be in the same order every time
             and the data that needs to be classified should be in the last row
         The table needs to be completley filled with only quantitative data
         The plot will only work with binary data
"""
    
    def __init__(self,col,est,ft,pca=False,n_components=5):
        """contructor for a general estimator class
        
            Args: 
                col(string): column to classify
                est(sklearn estimator): estimator use to classify
                ft(int): number of features to use
                pca(boolean): whether or not to use dimension reduction
                n_components(int): number of dimenstions to reduce to
        """
        self.estimator = est
        self.scale = preprocessing.MinMaxScaler()
        self.ft = ft
        self.pca = pca
        self.n_components = n_components
        self.plt_pca = PCA(n_components=2,random_state=0)
        self.col=col
    
    def fit(self,data):
        """fits your training data to your estimator
            Assumes the following is true: 
                The data is a pandas dataframe
                the column your trying to classify is the last column
                
                Args:
                    data(pandas dataframe): training data
        """
        pre_index = data.shape[1]-1
        pre_x, pre_y = data.values[:,:pre_index],data.values[:,pre_index]
        self.train_labels = pre_y
        
        labels = data.axes[1].values[:pre_index]
        self.features = Feature_selection.selectfeatures(pre_x,pre_y,self.ft,labels)
        data_ft = self.select_classifiers(data,self.features)
        self.index = data_ft.shape[1]-1
        x,y = data_ft.values[:,:self.index],data_ft.values[:,self.index]
        scaled_x = self.scale.fit_transform(x)
         
        if self.pca==True:
            self.dim_reducer = PCA(self.n_components)
            scaled_x = self.dim_reducer.fit_transform(scaled_x)
        
        self.createplot(scaled_x,y)
        self.estimator.fit(scaled_x,y)
        
        
    def select_classifiers(self,data,arr):
        """gets a new datatable from the columns specicfied in arr
        
            Args:
                data(pandas dataframe): table to condense
                arr(list of strings): col headers to keep in the table
                
            Returns:
                (pandas dataframe): Table with only columns from arr of data
        
        """
        dt = pd.DataFrame()
        for i in arr:
            dt.insert(dt.shape[1],i,data.get(i))
        dt.insert(dt.shape[1],self.col, data.get(self.col))
        return dt
    
    def check_model(self,data):
        """predicts and scores the test data using the estimator
            Args:
                data(pandas dataframe): test table to classify table from
            
            Returns:
                (list): list of the predicted classes
        """
        data_ft = self.select_classifiers(data,self.features)
        x, test_Y = data_ft.values[:,:self.index],data_ft.values[:,self.index]
        self.predict(x)
        test_labels = []
        for i in range(len(self.pred)):
            if self.pred[i]==test_Y[i]:
                if self.pred[i]==1:
                    test_labels.append('y')
                else:
                    test_labels.append('m')
            elif self.pred[i]-test_Y[i]==1:
                test_labels.append('b') #should have been 0 but guessed 1
            else:
                test_labels.append('r')  #should have been 1 but guessed 0
        self.addtoplot(self.scaled_x,test_labels)
        
        for i in range(len(self.pred)):
            print(self.pred[i],test_Y[i])
        self.f = f1_score(test_Y,self.pred)
        print("Accuracy:",1-sum(abs(self.pred-test_Y))/len(self.pred))
        print("f1 score:",self.f)
        return self.pred

    def predict(self,x):
        """predicts the classes for x based on the training data
            Args:
                x(numpy array): the data to be predicted by the model"""
        self.scaled_x = self.scale.transform(x)
        
        if self.pca==True:
            self.scaled_x = self.dim_reducer.transform(self.scaled_x)
        
        self.pred = self.estimator.predict(self.scaled_x)
    
    def createplot(self, data,labels):
        """Creates a plot and puts data on it 
            Args:
                data(pandas dataframe): data to be put on the plot
                labels(list): colors to lable the data
                
        """
        plt.figure(figsize = (10,8))
        Y = self.plt_pca.fit_transform(data)
        plt.scatter(Y[:,0], Y[:,1], 30, labels)
        
    def addtoplot(self, data, labels):
        """Puts more data on a plot 
            Args:
                data(pandas dataframe): data to be put on the plot
                labels(list): colors to lable the data
                
        """
        Y = self.plt_pca.transform(data)
        plt.scatter(Y[:,0], Y[:,1],50, labels, marker = "*")
        
    def plot(self):
        """plots the data -  currently works only for binary data"""
        class_colors = ["#ffff00","#483d8b","y","m","b","r"]
        classes = ["Given: 1","Given: 0","Correct: 1","Correct: 0","Wrong: guessed 1","Wrong: guessed 0"]
        recs = []
        for i in range(0,len(class_colors)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colors[i]))
        plt.legend(recs,classes,loc=2)
        plt.show()
    

    
def main():
    data_pd = pd.read_csv("/Users/baiintern1/Documents/Freesurfer_volume/brainscan_data/normal.csv")
    training_pd, testing_pd = model_selection.train_test_split(data_pd,test_size = 50, random_state=865)
    b = estimator("AV45 v1 Pos118",svm.SVC(kernel='rbf',C = 1048576, gamma = 3.0517578125e-05),25,True,13)
    b.fit(training_pd)
    b.check_model(testing_pd)
    b.plot()
    

        
        