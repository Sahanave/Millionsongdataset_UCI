
# coding: utf-8

# In[3]:

#import the libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from statistics import mode
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy 
from scipy import sparse
from sklearn.svm import LinearSVR 
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
#loading the dataset
f = open("/Users/sahanavenkatesh/Desktop/Semester-2 US/Machine Learning/Project/raw_data.txt")

dataset = numpy.loadtxt(f,delimiter=",")
train_dataset=dataset[0:463714,:]
test_dataset=dataset[463715:515344,:]
X_train=train_dataset[:,1:91]
X_trainsub=train_dataset[:,1:13]
y_train=train_dataset[:,0]
X_test=test_dataset[:,1:91]
X_testsub=test_dataset[:,1:13]
y_test=test_dataset[:,0]
scaler = preprocessing.StandardScaler().fit(train_dataset)
train_dataset=scaler.transform(train_dataset) 
train2_dataset=train_dataset[0:463714,:]
cross_val_data=train_dataset[363715:463714,:]
X_train=train2_dataset[:,1:91]
X_trainsub=train2_dataset[:,1:12]
y_train=train2_dataset[:,0]


#Model no 5 Extratrees Classifier
clf = RandomForestRegressor(n_estimators=20,criterion='mse', max_features='sqrt')
#Usually above 10 is good in most cases.
clf = clf.fit(X_train, y_train)
#################
  X_crossval=cross_val_data[:,1:91]
y_true=cross_val_data[:,0]
y_predict=clf.predict(X_crossval)
y_true=1998.3+y_true*10.93#scaling it back to measure the absolute error
y_predict=1998.3+y_predict*10.93#scaling it back to measure the absolute error
error_rf=mean_absolute_error(y_true, y_predict)
sq_error_rf=mean_squared_error(y_true, y_predict)
print('values for rf absolute error(years) and mean_squared_error(years)',error_rf)
print(sq_error_rf)






