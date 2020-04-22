# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:54:05 2020

@author: maxhi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

boston_dataset = load_boston()
dir(boston_dataset)
print(boston_dataset.DESCR)
print(boston_dataset.data.shape)
print(boston_dataset.feature_names)
print(boston_dataset.target)

df = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

df["Price"] = boston_dataset.target

print(df.head())
print(df.isnull().sum())
print(df.count())
print(df.info())


plt.hist(x=df["Price"],bins=40,ec="black")
plt.xlabel("Price in 000s")
plt.ylabel("Number of Houses")
plt.show()

sns.distplot(df["Price"],bins=40)
plt.show()

sns.distplot(df["RM"])
print(df["RM"].mean())


mask = np.zeros_like(df.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,10)) #Pearson corr only valid for continuous variables,
sns.heatmap(df.corr(),mask=mask, annot=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

nox_dis_corr = round(df["NOX"].corr(df["DIS"]),3)
plt.scatter(x=df["DIS"],y=df["NOX"],alpha=0.6,s=100)
plt.title(f'DIS vs NOX (Correlation {nox_dis_corr})')
plt.xlabel("DIS - Distance from employment")
plt.ylabel("NOX - Nitric Oxide Pollution")


sns.set()
sns.set_style("darkgrid")
sns.jointplot(x=df["DIS"],y=df["NOX"])
print(pearsonr(x=df["DIS"],y=df["NOX"]))

############MACHINE LEARNING#####################

prices = df["Price"]
prices_log = np.log(df["Price"])
features = df.drop("Price",axis=1)

X_train,X_test,y_train,y_test = train_test_split(features, prices, test_size=0.2,random_state=42)

sc = StandardScaler()
X_train_tr = sc.fit_transform(X_train) 
X_test_tr = sc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_tr,y_train)




coef_df = pd.DataFrame(data=lr.coef_,index=X_train.columns, columns=["coef"])
print(coef_df)

print("Training data r squared:", lr.score(X_train_tr,y_train))
print("Test data r squared:", lr.score(X_test_tr,y_test))

#Without log 74 train, 71 test //// With log 79 train, 76 test

##############Some STATS#######################
df["Price"].skew()
y_log = np.log(df["Price"])
print(y_log.skew())

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train,X_incl_const)
results = model.fit()
pd.DataFrame({"coef":results.params,"p-value": round(results.pvalues,3)})

#Testing for Multicollinearity
variance_inflation_factor(exog=X_incl_const.values,exog_idx=1)

vif = [] #Threshold about 10. Over 10 is problamatic
for i in range(X_incl_const.shape[1]):
    vif.append(variance_inflation_factor(exog=X_incl_const.values,exog_idx=i))
print(vif)

org_coef = pd.DataFrame({"coef_name":X_incl_const.columns,"vif":np.round(vif,2)})
print(results.bic) #-129
print(results.rsquared) #0.796

#Model complexity Basian Information Critirium
#Reduced model #1 exluding INUS
X_incl_const_without_inus = X_incl_const.drop(["INDUS"],axis=1)
model = sm.OLS(y_train,X_incl_const_without_inus)
results = model.fit()
coef_minus_indus  = pd.DataFrame({"coef_name":X_incl_const.columns,"vif":np.round(vif,2)})
print(results.bic) #-134
print(results.rsquared) #0.795


#Reducing by Indus, Age and ZN
X_incl_const_without_inus_age_zn = X_incl_const.drop(["INDUS","AGE","ZN"],axis=1)
model = sm.OLS(y_train,X_incl_const_without_inus_age_zn)
results = model.fit()
reduced_coef  = pd.DataFrame({"coef_name":X_incl_const.columns,"vif":np.round(vif,2)})
print(results.bic) #-144
print(results.rsquared) #0.794

######################### BACK TO ML #####################################

prices_log = np.log(df["Price"])
features = df.drop(["Price","INDUS","AGE","ZN"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(features, prices_log, test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

print("Training data r squared:", lr.score(X_train,y_train))
print("Test data r squared:", lr.score(X_test,y_test))

y_pred = lr.predict(X_test)
msq_lr = mean_squared_error(y_test,y_pred)
print(msq_lr)
print(np.exp(msq_lr))
print(np.sqrt(msq_lr))
print(lr.score(X_test,y_test))

############# RIDGE #################

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, features, prices_log, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)



def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

ridge = Ridge(normalize=True, alpha=0.2)
ridge.fit(X_train_tr,y_train)
ridge_pred = ridge.predict(X_test_tr)
print(mean_squared_error(y_test,ridge_pred))
print(np.sqrt(mean_squared_error(y_test,ridge_pred)))
print(ridge.score(X_test_tr,y_test))
