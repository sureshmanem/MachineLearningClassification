#!/usr/bin/env python
# coding: utf-8

# # Decision Tree: Clinical Dataset

# We will build a decision tree to predict the claim status based on drug details.

# ### Understanding and Cleaning the Data

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('Clinical_dataset.csv')


# In[4]:


# Type of values in each column of our dataframe 'df'.
df.info()


# In[5]:


# Understand the data, how it look like.
df.head()


# In[6]:


# missing values
round(100*(df.isnull().sum()/len(df)),2)


# In[7]:


# dropping RSTNDCLST and SMART_PA_SCH coulmns with null values
mdf = df.drop(columns=['RSTNDCLST','SMART_PA_SCH'])
mdf.head()


# In[8]:


mdf['GPIList_ID'] = mdf['List ID'] + '_' + mdf['PGO Generic Product ID']


# In[9]:


mdf.head()


# In[10]:


# dropping redundant List ID and PGO Generic Product ID columns
mdf = mdf.drop(columns=['List ID','PGO Generic Product ID'])
mdf.head()


# In[11]:


# clean dataframe
mdf.info()


# In[12]:


from sklearn import preprocessing


# encode categorical variables using Label Encoder

# select all categorical variables
mdf_categorical = mdf.select_dtypes(include=['object'])
mdf_categorical.head()


# In[13]:


# apply Label encoder to df_categorical

le = preprocessing.LabelEncoder()
mdf_categorical = mdf_categorical.apply(le.fit_transform)
mdf_categorical.head()


# In[14]:


# concat df_categorical with original df
mdf = mdf.drop(mdf_categorical.columns, axis=1)
mdf = pd.concat([mdf, mdf_categorical], axis=1)
mdf.head()


# In[15]:


# look at column types
mdf.info()


# In[16]:


# convert target variable status to categorical
mdf['Status'] = mdf['Status'].astype('category')


# Now all the categorical variables are suitably encoded. Let's build the model.

# <hr>

# ### Model Building and Evaluation

# Let's first build a decision tree with default hyperparameters. Then we'll use cross-validation to tune them.

# In[17]:


# Importing train-test-split 
from sklearn.model_selection import train_test_split


# In[18]:


# Putting feature variable to X
X = mdf.drop('Status',axis=1)

# Putting response variable to y
y = mdf['Status']


# In[19]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state = 99)
X_train.head()


# In[20]:


# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train, y_train)


# In[21]:


# Let's check the evaluation metrics of our default model

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classification report
print(classification_report(y_test, y_pred_default))


# In[22]:


# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))


# In[23]:


# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(mdf.columns[1:])
features


# In[24]:



import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz-2.38/release/bin/'


# In[25]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ### Hyperparameter Tuning

# ### Tuning max_depth

# In[26]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[27]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# Now let's visualize how train and test score changes with max_depth.

# In[28]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# You can see that as we increase the value of max_depth, both training and test score increase till about max-depth = 5, after which the test score gradually reduces. Note that the scores are average accuracies across the 5-folds. 

# ### Tuning min_samples_leaf

# In[29]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[30]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[31]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# You can see that at low values of min_samples_leaf, the tree gets a bit overfitted. At values > 25, however, the model becomes more stable and the training and test accuracy start to converge.

# <hr>

# ### Tuning min_samples_split

# In[32]:


# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[33]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[34]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# This shows that as you increase the min_samples_split, the tree overfits lesser since the model is less complex.

# <hr>

# ## Grid Search to Find Optimal Hyperparameters

# In[35]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(25, 150, 25),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[36]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[37]:


# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# <hr>

# **Running the model with best parameters obtained from grid search.**

# In[38]:


# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=25,
                                  min_samples_split=62)
clf_gini.fit(X_train, y_train)


# In[39]:


# accuracy score
clf_gini.score(X_test,y_test)


# In[40]:


# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# You can see that this tree is too complex to understand. Let's try reducing the max_depth and see how the tree looks.

# In[41]:


# tree with max_depth = 3
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=25,
                                  min_samples_split=62)
clf_gini.fit(X_train, y_train)

# score
print(clf_gini.score(X_test,y_test))


# In[42]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[43]:


# classification metrics
from sklearn.metrics import classification_report,confusion_matrix
y_pred = clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))


# In[44]:


# confusion matrix
print(confusion_matrix(y_test,y_pred))


# <hr>
