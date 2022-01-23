#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, ConfusionMatrixDisplay


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['Geography'].unique()


# In[6]:


gd = pd.get_dummies(data=df['Geography'])


# In[7]:


df = df.join(gd)


# In[8]:


df = df.drop(['RowNumber','Geography'], axis=1)


# In[9]:


df['Gender'] = df['Gender'].replace({'Female':0, 'Male':1})


# In[10]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)


# In[11]:


df.head()


# In[12]:


X = df.drop(['CustomerId','Surname', 'Exited'],axis=1)


# In[13]:


X.shape


# In[14]:


y = df['Exited']


# In[15]:


y.value_counts().plot(kind='bar')


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)


# In[17]:


X_train.shape


# In[18]:


X_test.shape


# In[19]:


scaler = StandardScaler()


# In[20]:


scaled_X_train = scaler.fit_transform(X_train)


# In[21]:


scaled_X_test = scaler.transform(X_test)


# ### Logistic Regression

# In[22]:


log_model = LogisticRegression(solver='saga', penalty='l1', C=1)


# In[23]:


# {'C': 1.0, 'l1_ratio': 0.0, 'penalty': 'l1', 'solver': 'saga'}


# In[24]:


# solver = ['lbfgs', 'liblinear', 'sag', 'saga']
# penalty = ['l1', 'l2', 'elasticnet']
# C = np.logspace(0,10,10)
# l1_ratio = np.linspace(0,1,10)
# log_param_grid = {'solver':solver, 'penalty':penalty, 'C':C, 'l1_ratio':l1_ratio }


# In[25]:


# log_grid_model = GridSearchCV(log_model, log_param_grid, cv=5, n_jobs=-1)

# n_jobs= -1 bütün işlemcileri çalıştırıyor.


# In[26]:


log_model.fit(scaled_X_train,y_train)


# In[27]:


# log_grid_model.best_params_


# In[28]:


y_pred = log_model.predict(scaled_X_test)


# In[29]:


accuracy_score(y_test, y_pred)


# In[30]:


recall_score(y_test, y_pred)


# In[31]:


precision_score(y_test, y_pred)


# In[32]:


confusion_matrix(y_test,y_pred)


# In[33]:


print(classification_report(y_test, y_pred))


# In[34]:


log_model_report = classification_report(y_test, y_pred)


# ### KNN

# In[35]:


knn_model = KNeighborsClassifier(algorithm='ball_tree', metric='manhattan', n_neighbors=8, weights='distance', n_jobs=-1)


# In[36]:


# {'algorithm': 'ball_tree', 'metric': 'manhattan', 'n_neighbors': 8, 'weights': 'distance'}


# In[37]:


# knn_model.get_params().keys()


# In[38]:


# n_neighbors = list(range(1,11))
# algorithm = ['ball_tree','kd_tree','brute']
# metric = ['minkowski','manhattan','euclidean']
# weights= ['uniform','distance']


# In[39]:


#param_grid = {'n_neighbors':n_neighbors,
#              'algorithm':algorithm,
#              'metric': metric, 
#              'weights': weights}


# In[40]:


# knn_grid_model = GridSearchCV(knn_model, param_grid,cv=5, n_jobs=-1)


# In[41]:


knn_model.fit(scaled_X_train, y_train)


# In[42]:


# knn_grid_model.best_params_


# In[43]:


y_pred = knn_model.predict(scaled_X_test)


# In[44]:


confusion_matrix(y_test,y_pred)


# In[45]:


print(classification_report(y_test, y_pred))


# In[46]:


knn_model_report = classification_report(y_test, y_pred)


# ### SVM

# In[47]:


svc_model = SVC(C=1, kernel='poly')


# In[48]:


# svc_model.get_params().keys()


# In[49]:


# C = [0.01,0.1,1]
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']


# In[50]:


# param_grid = {'C':C, 'kernel':kernel}


# In[51]:


# svc_grid_model = GridSearchCV(svc_model, param_grid,cv=5, n_jobs=-1)


# In[52]:


svc_model.fit(scaled_X_train, y_train)


# In[53]:


# {'C': 1, 'kernel': 'poly'}


# In[54]:


# svc_grid_model.best_params_


# In[55]:


y_pred = svc_model.predict(scaled_X_test)


# In[56]:


confusion_matrix(y_test,y_pred)


# In[57]:


print(classification_report(y_test, y_pred))


# In[58]:


svc_model_report = classification_report(y_test, y_pred)


# ### Decision Tree

# In[59]:


tree_model = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=10,min_impurity_decrease=0,min_samples_leaf=1,min_samples_split=2)


# In[60]:


# {'max_depth': 5, 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2}


# In[61]:


#param_grid = {"max_depth": range(1,11),
#              "min_samples_split": range(2,11,2) ,
#              "min_samples_leaf":range(1,11),
#              "max_leaf_nodes": range(2,11) ,
#              "min_impurity_decrease": np.arange(0,1,0.1)
#             }


# In[62]:


# tree_grid_model = GridSearchCV(tree_model,param_grid,cv=5, n_jobs=-1)


# In[63]:


tree_model.fit(scaled_X_train,y_train)


# In[64]:


# tree_grid_model.best_params_


# In[65]:


plt.bar(range(len(tree_model.feature_importances_)), tree_model.feature_importances_)


# In[66]:


y_pred = tree_model.predict(scaled_X_test)


# In[67]:


confusion_matrix(y_test,y_pred)


# In[68]:


print(classification_report(y_test, y_pred))


# In[69]:


tree_model_report = classification_report(y_test, y_pred)


# ### Random Forest

# In[70]:


rfc_model = RandomForestClassifier(bootstrap=True, max_depth=10, n_estimators=100)


# In[71]:


# {'bootstrap': True, 'max_depth': 10, 'n_estimators': 100}


# In[72]:


#param_grid = {"max_depth": range(1,11),
#              "n_estimators": [64,100,128,200],
#              "bootstrap": [True, False]
#             }


# In[73]:


# rfc_grid_model = GridSearchCV(rfc_model,param_grid,cv=5, n_jobs=-1)


# In[74]:


rfc_model.fit(scaled_X_train,y_train)


# In[75]:


# rfc_grid_model.best_params_


# In[76]:


y_pred = rfc_model.predict(scaled_X_test)


# In[77]:


rfc_model.feature_importances_


# In[78]:


plt.bar(range(len(rfc_model.feature_importances_)), rfc_model.feature_importances_)


# In[79]:


confusion_matrix(y_test,y_pred)


# In[80]:


print(classification_report(y_test, y_pred))


# In[81]:


rfc_model_report = classification_report(y_test, y_pred)


# In[82]:


print(log_model_report, knn_model_report, svc_model_report, tree_model_report, rfc_model_report)


# ## XGBoost

# In[83]:


import xgboost as xgb


# In[84]:


xgb_model = xgb.XGBClassifier(eval_metric=f1_score, use_label_encoder=False)


# In[85]:


xgb_model.fit(scaled_X_train,y_train)


# In[86]:


xgb_model.feature_importances_


# In[87]:


plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)


# In[88]:


xgb.plot_importance(xgb_model)


# In[89]:


y_pred = xgb_model.predict(scaled_X_test)


# In[90]:


xgb_model_report = classification_report(y_test, y_pred)


# In[91]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[113]:


xgb_matrix = xgb.DMatrix(data=X, label=y)

params = {
    'max_depth': 3,
    'learning_rate': 0.5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1
}


# In[114]:


xgb_cv = xgb.cv(
    dtrain=xgb_matrix,
    params = params,
    num_boost_round=50,
    nfold=5,
    metrics=('logloss'),
    early_stopping_rounds=10,
    verbose_eval=True,    
)


# In[115]:


xgb_train = xgb.train(params=params, dtrain=xgb_matrix, num_boost_round=10)


# In[116]:


xgb.plot_tree(xgb_train)


# In[127]:


xgb.plot_importance(xgb_train)


# ### objective [default=reg:squarederror]
# 
# **reg:squarederror:** regression with squared loss.
# 
# **reg:squaredlogerror**: regression with squared log loss 
#  
# . All input labels are required to be greater than -1. Also, see metric rmsle for possible issue with this objective.
# 
# **reg:logistic:** logistic regression
# 
# **reg:pseudohubererror:** regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
# 
# **binary:logistic:** logistic regression for binary classification, output probability
# 
# **binary:logitraw:** logistic regression for binary classification, output score before logistic transformation
# 
# **binary:hinge:** hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
# 
# **count:poisson** –poisson regression for count data, output mean of Poisson distribution
# 
# max_delta_step is set to 0.7 by default in Poisson regression (used to safeguard optimization)
# 
# **survival:cox:** Cox regression for right censored survival time data (negative values are considered right censored). Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function h(t) = h0(t) * HR).
# 
# **survival:aft:** Accelerated failure time model for censored survival time data. See Survival Analysis with Accelerated Failure Time for details.
# 
# **aft_loss_distribution:** Probability Density Function used by survival:aft objective and aft-nloglik metric.
# 
# **multi:softmax:** set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
# 
# **multi:softprob:** same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class.
# 
# **rank:pairwise:** Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
# 
# **rank:ndcg:** Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized
# 
# **rank:map:** Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is maximized
# 
# **reg:gamma:** gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed.
# 
# **reg:tweedie:** Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.

# ### tree_method
# 
# **auto:** Use heuristic to choose the fastest method.
# 
# For small dataset, exact greedy (exact) will be used.
# 
# For larger dataset, approximate algorithm (approx) will be chosen. It’s recommended to try hist and gpu_hist for higher performance with large dataset. (gpu_hist)has support for external memory.
# 
# Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
# 
# **exact:** Exact greedy algorithm. Enumerates all split candidates.
# 
# **approx:** Approximate greedy algorithm using quantile sketch and gradient histogram.
# 
# **hist:** Faster histogram optimized approximate greedy algorithm.
# 
# **gpu_hist:** GPU implementation of hist algorithm.

# In[156]:


param_grid = {
    #'booster': ['gbtree','dart'],
    #'tree_method': ['exact','hist'],
    #'colsample_bytree': [0,0.5],
    #'colsample_bylevel': [0,0.5],
    #'colsample_bynode': [0,0.5],
    #'max_delta_step': [0,1],
    #'max_depth': [10,100],
    'gamma': [1,2],
    'reg_alpha':[5,6],
    'reg_lambda': [1,2],
    'n_jobs': [-1]  
}


# In[157]:


xgb_grid = GridSearchCV(
    xgb_model,
    param_grid,
    scoring='f1',
    n_jobs=-1,
    cv = 10,
    verbose=10
)


# In[158]:


xgb_grid.fit(scaled_X_train,y_train)


# In[159]:


xgb_grid.best_params_


# In[160]:


y_pred = xgb_grid.predict(scaled_X_test)


# In[163]:


print(classification_report(y_test, y_pred))


# In[162]:


print(xgb_model_report)


# # Imblearn

# ## RandomOverSampler

# In[ ]:


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTEN, SMOTENC, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE


# In[ ]:


ros = RandomOverSampler(sampling_strategy=1,random_state=1,shrinkage=1)


# In[ ]:


ros.get_params()


# In[ ]:


X_resampled, y_resampled = ros.fit_resample(scaled_X_train, y_train)


# In[ ]:


from collections import Counter
print(sorted(Counter(y_resampled).items()))


# In[ ]:


y_resampled.value_counts()


# In[ ]:


y_train.value_counts()


# RandomOverSampler, az sayıdaki örnekleri çoğaltarak eşitlik sağlıyor. shrinkage değeri verilmezse örnekleri duplicate eder. shrinkage değeri verilirse duplicate etmez. Mevcut verilere çok yakın değerler üretir

# In[ ]:


log_model.fit(X_resampled,y_resampled)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


# before resampled
print(log_model_report)


# In[ ]:


# after resampled
print(classification_report(y_test, y_pred))


# ### SMOTE - Synthetic Minority Oversampliing Technique

# SMOTE might connect inliers and outliers

# In[ ]:


X_smote, y_smote = SMOTE(k_neighbors=5, n_jobs=-1).fit_resample(scaled_X_train, y_train)


# In[ ]:


log_model.fit(X_smote,y_smote)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# SMOTE offers three additional options to generate samples. Those methods focus on samples near the border of the optimal decision function and will generate samples in the opposite direction of the nearest neighbors class. 

# #### BorderlineSMOTE, SVMSMOTE, KMeansSMOTE

# In[ ]:


BorderlineSMOTE().get_params()


# In[ ]:


SVMSMOTE().get_params() 

# svm_estimator default SVC
# out_step: step size
# m_neighbors : number of nearest neighbours to use to determine if a minority sample is in danger. 


# In[ ]:


KMeansSMOTE().get_params()

# applying a clustering before to oversample


# #### SMOTENC, SMOTEN

# In[ ]:


SMOTENC(categorical_features=df['Surname']).get_params()

# only working when data is a mixed of numerical and categorical features. 


# In[ ]:


SMOTEN().get_params()

# If data are made of only categorical data, one can use the SMOTEN variant


# ### ADASYN - Adaptive Synthetic

# ADASYN might focus solely on outliers

# In[ ]:


X_adasyn, y_adasyn = ADASYN(n_neighbors=5, n_jobs=-1).fit_resample(scaled_X_train, y_train)


# In[ ]:


log_model.fit(X_adasyn,y_adasyn)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# While the RandomOverSampler is over-sampling by duplicating some of the original samples of the minority class, SMOTE and ADASYN generate new samples in by interpolation. ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier while the basic implementation of SMOTE will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule. Therefore, the decision function found during training will be different among the algorithms.

# ### 1.Prototype Selection

# prototype selection algorithms will select samples from the original set S. these algorithms can be divided into two groups: **(i) the controlled under-sampling techniques and (ii) the cleaning under-sampling techniques**. The first group of methods allows for an under-sampling strategy in which the number of samples in S' is specified by the user. By contrast, cleaning under-sampling techniques do not allow this specification and are meant for cleaning the feature space.

# ### a.Controlled under-sampling techniques

# #### RandomUnderSampler

# RandomUnderSampler randomly deletes the rows of the majority class(es) according to our sampling strategy.

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours


# In[ ]:


rus = RandomUnderSampler(replacement=True)


# In[ ]:


rus.get_params()


# In[ ]:


X_undersampled, y_undersampled = rus.fit_resample(scaled_X_train, y_train)


# In[ ]:


print(sorted(Counter(y_undersampled).items()))


# In[ ]:


y_undersampled.value_counts()


# In[ ]:


y_train.value_counts()


# In[ ]:


log_model.fit(X_undersampled,y_undersampled)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# #### NearMiss

# NearMiss heuristic rules are based on nearest neighbors algorithm. Therefore, the parameters n_neighbors and n_neighbors_ver3 accept classifier derived from KNeighborsMixin from scikit-learn. The former parameter is used to compute the average distance to the neighbors while the latter is used for the pre-selection of the samples of interest.

# In[ ]:


near = NearMiss(version=3)


# In[ ]:


near.get_params()


# **NearMiss-1** selects samples from the majority class for which the average distance of the k nearest samples of the minority class is the smallest.  
# **NearMiss-2** selects the samples from the majority class for which the average distance to the farthest samples of the negative class is the smallest. 
# **NearMiss-3** is a 2-step algorithm: first, for each minority sample, their m nearest-neighbors will be kept; then, the majority samples selected are the on for which the average distance to the k nearest neighbors is the largest.

# In[ ]:


X_near, y_near = near.fit_resample(scaled_X_train,y_train) 


# In[ ]:


y_near.value_counts()


# In[ ]:


y_train.value_counts()


# In[ ]:


log_model.fit(X_near,y_near)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# ### b.Cleaning under-sampling techniques

# #### Tomek's Link

# a Tomek’s link exist if the two samples are the nearest neighbors of each other. The parameter sampling_strategy control which sample of the link will be removed. For instance, the default (i.e., sampling_strategy='auto') will remove the sample from the majority class. Both samples from the majority and minority class can be removed by setting sampling_strategy to 'all'. 

# In[ ]:


tomek = TomekLinks(sampling_strategy='auto')


# In[ ]:


tomek.get_params()


# In[ ]:


X_tomek, y_tomek = tomek.fit_resample(scaled_X_train,y_train)


# In[ ]:


y_tomek.value_counts()


# In[ ]:


y_train.value_counts()


# #### EditedNearestNeighbours

# EditedNearestNeighbours applies a nearest-neighbors algorithm and “edit” the dataset by removing samples which do not agree “enough” with their neighboorhood. For each sample in the class to be under-sampled, the nearest-neighbours are computed and if the selection criterion is not fulfilled, the sample is removed. Two selection criteria are currently available: (i) the majority (i.e., kind_sel='mode') or (ii) all (i.e., kind_sel='all') the nearest-neighbors have to belong to the same class than the sample inspected to keep it in the dataset. Thus, it implies that kind_sel='all' will be less conservative than kind_sel='mode', and more samples will be excluded in the former strategy than the latest:

# In[ ]:


enn = EditedNearestNeighbours(kind_sel='all')


# In[ ]:


enn.get_params()


# In[ ]:


X_enn, y_enn = enn.fit_resample(scaled_X_train, y_train)


# In[ ]:


y_enn.value_counts()


# In[ ]:


y_train.value_counts()


# #### RepeatedEditedNearestNeighbours

# RepeatedEditedNearestNeighbours extends EditedNearestNeighbours by repeating the algorithm multiple times. Generally, repeating the algorithm will delete more data:

# In[ ]:


renn = RepeatedEditedNearestNeighbours(max_iter=3)


# In[ ]:


renn.get_params()


# In[ ]:


X_renn, y_renn = renn.fit_resample(scaled_X_train,y_train)


# In[ ]:


y_renn.value_counts()


# In[ ]:


y_train.value_counts()


# ### 2.Prototype generation

# Prototype generation technique will reduce the number of samples in the targeted classes but the remaining samples are **generated — and not selected — from the original set.**

# #### ClusterCentroids

# use of K-means to reduce the number of samples. Therefore, each class will be synthesized with the centroids of the K-means method instead of the original samples:

# In[ ]:


cc = ClusterCentroids()


# In[ ]:


cc.get_params()


# In[ ]:


X_cc, y_cc = cc.fit_resample(scaled_X_train,y_train)


# In[ ]:


y_cc.value_counts()


# In[ ]:


y_train.value_counts()


# In[ ]:


log_model.fit(X_cc,y_cc)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# ## Combination of over- and under-sampling

# #### SMOTEENN SMOTETomek

# In[ ]:


from imblearn.combine import SMOTEENN,SMOTETomek


# In[ ]:


smoteenn = SMOTEENN()

# Over-sampling using SMOTE and cleaning using ENN(Edited Nearest Neighbours)


# In[ ]:


smoteenn.get_params()


# In[ ]:


X_smoteenn, y_smoteenn = smoteenn.fit_resample(scaled_X_train, y_train)


# In[ ]:


y_smoteenn.value_counts()


# In[ ]:


log_model.fit(X_smoteenn,y_smoteenn)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


somek = SMOTETomek()

# Over-sampling using SMOTE and cleaning using Tomek links.


# In[ ]:


somek.get_params()


# In[ ]:


X_somek, y_somek = somek.fit_resample(scaled_X_train, y_train)


# In[ ]:


y_somek.value_counts()


# In[ ]:


log_model.fit(X_somek, y_somek)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# ### Pipeline

# In[ ]:


from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score


# In[ ]:


over = RandomOverSampler(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.6)


# In[ ]:


# over: raises the minority class to “0.5 * majority class”
# under: reduces the majority class quantity to “0.6 * minority class”


# In[ ]:


steps = [('o',over),('u',under)]


# In[ ]:


pipeline = Pipeline(steps=steps)


# In[ ]:


X_com, y_com = pipeline.fit_resample(scaled_X_train, y_train)


# In[ ]:


y_com.value_counts()


# In[ ]:


y_train.value_counts()


# In[ ]:


log_model.fit(X_com,y_com)


# In[ ]:


y_pred = log_model.predict(scaled_X_test)


# In[ ]:


f1_score()


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


for i in np.arange(0.1,1,0.1):
    for j in np.arange(0.1,1,0.1):
        try:
            over = RandomOverSampler(sampling_strategy=i)
            under = RandomUnderSampler(sampling_strategy=j)
            steps = [('o',over),('u',under)]
            pipeline = Pipeline(steps=steps)
            X_com, y_com = pipeline.fit_resample(scaled_X_train, y_train)
            log_model.fit(X_com,y_com)
            y_pred = log_model.predict(scaled_X_test)
            print('over: ',i,'under:', j, 'f1_score:',f1_score(y_test,y_pred))
        except ValueError:
            pass


# In[ ]:


over = RandomOverSampler(sampling_strategy=0.6)
under = RandomUnderSampler(sampling_strategy=0.7)


# In[ ]:


steps = [('o',over),('u',under)]
pipeline = Pipeline(steps=steps)
X_last, y_last = pipeline.fit_resample(scaled_X_train, y_train)
X_deneme, y_deneme = pipeline.fit_resample(X_train, y_train)


# #### logistic regression

# In[ ]:


model = LogisticRegression()


# In[ ]:


# model.get_params()


# In[ ]:


param_grid = {
    'n_jobs':[-1],
    'C':[0.01,0.5,1],
    'penalty': ['elasticnet', 'l1', 'l2', 'none'],
    'l1_ratio': [0,0.5,1],
    'solver': ['saga']
             }


# In[ ]:


grid = GridSearchCV(model, param_grid, n_jobs=-1, cv=10, return_train_score=True, scoring='f1')


# In[ ]:


grid.fit(X_last,y_last)


# In[ ]:


# grid.best_params_ : {'C': 1, 'n_jobs': -1, 'solver': 'saga'}
# grid.best_score_ : 0.7146690917189413
# best_params ile yeniden grid.


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# #### KNN

# In[ ]:


model = KNeighborsClassifier()


# In[ ]:


# model.get_params(())


# In[ ]:


param_grid = {
    'n_jobs': [-1],
    'n_neighbors': [3,5],
    'metric': ['manhattan','minkowski', 'euclidean']
}


# In[ ]:


grid = GridSearchCV(model,param_grid,n_jobs=-1,cv=10, return_train_score=True, scoring='f1')


# In[ ]:


grid.fit(X_last,y_last)


# In[ ]:


# grid.best_params_ : {'algorithm': 'brute', 'metric': 'manhattan', 'n_jobs': -1, 'n_neighbors': 5}
# grid.best_score_ : 0.7892138144463828


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# KNN is better than logistic regression


# #### SVM

# In[ ]:


model = SVC()


# In[ ]:


# model.get_params()


# In[ ]:


param_grid = {
    'kernel': ['rbf'],
    'C': [0.1,1],
    'gamma': ['scale']
}


# In[ ]:


grid = GridSearchCV(model,param_grid, cv=10, n_jobs=-1, return_train_score=True, scoring='f1')


# In[ ]:


grid.fit(X_last,y_last)


# In[ ]:


# grid.best_params_ : {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
# grid.best_score_ : 0.8012718765079248


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# SVM is better than KNN


# #### Decision Tree

# In[ ]:


model = DecisionTreeClassifier()


# In[ ]:


# model.get_params()


# In[ ]:


param_grid = {
    'max_depth': [100],
    'min_samples_split': [10],
}


# In[ ]:


grid = GridSearchCV(model, param_grid, cv=10, n_jobs=-1, return_train_score=True, scoring='f1')


# In[ ]:


grid.fit(X_last,y_last)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# #### Random Forest

# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.get_params()


# In[ ]:


param_grid = {
    'bootstrap': [True,False],
    'n_jobs': [-1],
    'n_estimators': [100,1000],
    'min_samples_leaf': [1,5,10]
}


# In[ ]:


grid = GridSearchCV(model, param_grid, cv=10, n_jobs=-1, return_train_score=True, scoring='f1')


# In[ ]:


grid.fit(X_last,y_last)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# #### Balance with scaled

# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# close to SVM


# #### First balance then scale

# In[ ]:


X_deneme_scaled = scaler.fit_transform(X_deneme)


# In[ ]:


grid.fit(X_deneme_scaled,y_deneme)


# In[ ]:


y_pred = grid.predict(scaled_X_test)


# In[ ]:


print(classification_report(y_test,y_pred))

