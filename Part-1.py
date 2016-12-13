
# coding: utf-8

# In[109]:

#import the libraries
import pandas
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
from sklearn.ensemble import ExtraTreesRegressor



# In[110]:

#loading the dataset
f = open("/Users/sahanavenkatesh/Desktop/Semester-2 US/Machine Learning/Project/raw_data.txt")


# In[111]:

dataset = numpy.loadtxt(f,delimiter=",")


# In[112]:


train_dataset=dataset[0:463714,:]
test_dataset=dataset[463715:515344,:]
X_train=train_dataset[:,1:91]
y_train=train_dataset[:,0]
X_test=test_dataset[:,1:91]
y_test=test_dataset[:,0]
#the features are the 12 timbre averages and their covariances across segements for each song


# In[113]:

#stats
#understanding the distribution of the year

mean_years=numpy.mean(train_dataset[:,0])
#mean of the year
standard_year=numpy.std(train_dataset[:,0])
#standard deviation
#minimum and the maximum year
y_min=min(train_dataset[:,0])
y_max=max(train_dataset[:,0])
print ("mean year is ",mean_years)
print ("the range of the years is from",y_min) 
print("to",y_max)
print ("the standard deviation of the years is ",standard_year)
modey=mode(train_dataset[:,0])
print('the mode is',modey)

    
            

          


# In[114]:

#plotting the year distribution
plt.hist(train_dataset[:,0])
plt.title("Distribution of years")
plt.xlabel("years")
plt.ylabel("Frequency")
plt.show()


# In[115]:

scaler = preprocessing.StandardScaler().fit(train_dataset)
train_dataset=scaler.transform(train_dataset) 
train2_dataset=train_dataset[0:463714,:]
cross_val_data=train_dataset[363715:463714,:]
X_train=train2_dataset[:,1:91]
X_trainsub=train2_dataset[:,1:12]
y_train=train2_dataset[:,0]
print('Is my matrix with 90 columns sparse')
print (sparse.issparse(X_train))
print('Is my matrix with 12 PCA components as columns sparse')
print (sparse.issparse(X_trainsub))


# In[116]:

#model no 1
#performing ridge regression by taking all features 
model = Ridge()
alphas = numpy.array([1,0.1,0.01,0.001,0.0001])
f_scorer=make_scorer(mean_squared_error,greater_is_better=False)
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),scoring=f_scorer,cv=10)

grid.fit(X_train, y_train)
# summarize the results of the grid search
alpha_ridge=grid.best_estimator_.alpha
clf_ridge = SGDRegressor(loss='squared_loss', penalty='l2', alpha=alpha_ridge, l1_ratio=0)
clf_ridge.fit(X_train, y_train)
weights_ridge=clf_ridge.coef_
#Validating the model by calculating the error in the cross validation set
X_crossval=cross_val_data[:,1:91]
y_true=cross_val_data[:,0]
y_predict=clf_ridge.predict(X_crossval)
y_true=1998.3+y_true*10.93#scaling it back to measure the absolute error
y_predict=1998.3+y_predict*10.93#scaling it back to measure the absolute error
error_ridge=mean_absolute_error(y_true, y_predict)
sq_error_ridge=mean_squared_error(y_true, y_predict)
print('values for ridge absolute error(years) and mean_squared_error(years)',error_ridge)
print(sq_error_ridge)


# In[117]:

#model no 2 Performing elastic net on 90 features
model =  ElasticNet()
alphas=numpy.array([1,0.1,0.01,0.001,0.0001])
f_score=make_scorer(mean_squared_error,greater_is_better=False)
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),scoring=f_score,cv=10)
grid.fit(X_train, y_train)
# summarize the results of the grid search
alpha_net=grid.best_estimator_.alpha
clf_net = SGDRegressor(loss='squared_loss', penalty='elasticnet', alpha=alpha_net, l1_ratio=0.15)
clf_net.fit(X_train, y_train)
weights_net=clf_net.coef_
X_crossval=cross_val_data[:,1:91]
y_true=cross_val_data[:,0]
y_predict=clf_net.predict(X_crossval)
y_true=1998.3+y_true*10.93#scaling it back to measure the absolute error
y_predict=1998.3+y_predict*10.93#scaling it back to measure the absolute error
error_net=mean_absolute_error(y_true, y_predict)
sq_error_net=mean_squared_error(y_true, y_predict)
print('values for Elasticnet absolute error(years) and mean_squared_error(years)',error_net)
print(sq_error_net) 


# In[118]:

#model no 3
#performing lasso by taking the first 12 features
model = Lasso()
alphas = numpy.array([1,0.1,0.01,0.001,0.0001])
f_scorer=make_scorer(mean_squared_error,greater_is_better=False)
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),scoring=f_scorer,cv=10)
grid.fit(X_train[:,0:12], y_train)
# summarize the results of the grid search
alpha_lasso=grid.best_estimator_.alpha
clf_lasso = SGDRegressor(loss='squared_loss', penalty='l1', alpha=alpha_lasso, l1_ratio=1)
clf_lasso.fit(X_train[:,0:12], y_train)
weights_lasso=clf_lasso.coef_
X_crossval=cross_val_data[:,0:12]
y_true=cross_val_data[:,0]
y_predict=clf_lasso.predict(X_crossval)
y_true=1998.3+y_true*10.93#scaling it back to measure the absolute error
y_predict=1998.3+y_predict*10.93#scaling it back to measure the absolute error
error_lasso=mean_absolute_error(y_true, y_predict)
sq_error_lasso=mean_squared_error(y_true, y_predict)
print('values for lasso absolute error(years) and mean_squared_error(years)',error_lasso)
print(sq_error_lasso)  





# In[119]:

features=range(0,90)
plt.gca().set_color_cycle(['red', 'green', 'black'])
plt.plot(features,weights_ridge)
plt.plot(features,weights_net)
plt.plot(features[0:12],weights_lasso)
plt.legend()
plt.title('Weights of the Features')
plt.legend(['ridge', 'Elastic Net', 'Lasso'], loc='upper left')
plt.show()


# In[120]:

#training Elasticnet on the entire test data and testing it on the test data
model =  ElasticNet()
alphas=numpy.array([1,0.1,0.01,0.001,0.0001])
f_score=make_scorer(mean_squared_error,greater_is_better=False)
grid = GridSearchCV(estimator=model,param_grid=dict(alpha=alphas),scoring=f_score,cv=10)
X_train=train_dataset[:,1:91]
grid.fit(X_train, y_train)
# summarize the results of the grid search
alpha_net=grid.best_estimator_.alpha
clf_net = SGDRegressor(loss='squared_loss', penalty='elasticnet', alpha=alpha_net, l1_ratio=0.15)
clf_net.fit(X_train, y_train)
weights_net=clf_net.coef_
test_dataset=dataset[463715:515344,:]
test_dataset=scaler.transform(test_dataset)
X_test=test_dataset[:,1:91]
y_test=test_dataset[:,0]
y_predict=clf_net.predict(X_test)
y_test=1998.3+y_test*10.93#scaling it back to measure the absolute error
y_predict=1998.3+y_predict*10.93#scaling it back to measure the absolute error
error_net=mean_absolute_error(y_test, y_predict)
sq_error_net=mean_squared_error(y_test, y_predict)
print('values for Elasticnet absolute error(years) for test set and mean_squared_error(years)for test set',error_net)
print(sq_error_net) 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



