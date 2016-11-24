# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:39:05 2016

@author: 
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.set_printoptions(precision=4, suppress = True)

Trained_NormalizedFeature = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\TestedFeature\Training_Norm_Data.npy')
Trained_NormalizedFeature_Female = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\TestedFeature\Training_Norm_Data_Female.npy')
Trained_NormalizedFeature_Mixed = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\TestedFeature\Training_Norm_Data_Mixed.npy')
Trained_NormalizedFeature_Complete = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\TestedFeature\Training_Norm_Data_Complete.npy')

Testing_Norm_data = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data.npy')
Testing_Norm_data_Female = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Female.npy')
Testing_Norm_data_Female2 = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Female2.npy')
Testing_Norm_data_Female_Complete = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Female_Complete.npy')
Testing_Norm_data_Mixed = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Mixed.npy')
Testing_Norm_data_Mixed2 = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Mixed2.npy')
Testing_Norm_data_Complete = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Complete.npy')
Testing_Norm_data_Complete2 = np.load('C:\Users\IBM_ADMIN\Documents\Python Scripts\SingleFileTesting\Testing_Norm_Data_Complete2.npy')

clf = SVC(probability=True)
y = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,\
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,\
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,\
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,\
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,\
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,\
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,\
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
clf.fit(Trained_NormalizedFeature_Complete, y) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    

#prediction
true_output = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
               2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,\
               3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
output=clf.predict(Testing_Norm_data_Complete2)
target_names = ['Angry', 'Calm', 'Happy']
print 'Accuracy :', accuracy_score(true_output,output)
print 'Confusion Matrix :\n',confusion_matrix(true_output,output)
print 'Classification Report :\n', (classification_report(true_output, output, target_names=target_names))
Probabilities = clf.predict_proba(Testing_Norm_data_Complete2)
#print 'Probabilities:', Probabilities
#print 'Predicted class:', output

def singleSampleTesting():
    return Probabilities

