# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:37:02 2020

@author: maxhi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("original.csv")

print(data.columns)

print(data.isnull().sum())

df = data.drop(["Rank","Release Date","Movie Title","Domestic Gross ($)"],axis=1)
print(df.dtypes)

#Clean the $ signs

df["prod_budget"] = df["Production Budget ($)"].str.replace(",","").str.replace("$","")
df["ww_gross"] = df["Worldwide Gross ($)"].str.replace(",","").str.replace("$","")
df["ww_gross"] = df["ww_gross"].astype("int64")
df["prod_budget"] = df["prod_budget"].astype("int64")

df = df.drop(["Worldwide Gross ($)","Production Budget ($)"],axis=1)

#Check the 0 Values

df = df[(df["prod_budget"] > 0) & (df["ww_gross"] > 0)]

print(df.describe())


X = df["prod_budget"].values
y = df["ww_gross"].values

plt.figure(figsize=(10,6))
plt.scatter(X,y)
plt.title("Film Cost vs Global Revenue")
plt.xlabel("Production budget in $")
plt.ylabel("World Wide Gross in $")
plt.ylim(0, 3000000000)
plt.xlim(0,450000000)
plt.show()

