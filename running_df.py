#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 08:52:29 2017

@author: diego

references:
    http://scikit-learn.org/stable/modules/multiclass.html
    https://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
"""
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score

# Get data as dataframe
dfData = pd.read_csv('noticias_op_pol.csv')
# Get a sample of the dataframe
df = dfData.sample(frac=0.7)

# Transform column from dataframe
newsList = df['news'].tolist()
# Convert to array
newsArray = np.asarray(newsList)

# Transform category column to list
categoryList = df['category'].tolist()
# Convert to array
categoryArray = np.asarray(categoryList)

# Get 60/20/20 of data
# Train
trainNewsArray = newsArray[0:int(len(newsArray)*0.6)]
trainCategoryArray = categoryArray[0:int(len(categoryArray)*0.6)]

#Test
testNewsArray = newsArray[int(len(newsArray)*0.4):]
testCategoryArray = categoryArray[int(len(categoryArray)*0.4):]

# Target names
targetNames = ['POLÍTICA', 'OPINIÃO']

# Training classifier
classifier = Pipeline([
        #('vectorizer', CountVectorizer(min_n = 1, max_n = 2)),
        #('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))
        ]).fit(trainNewsArray, trainCategoryArray)

prdct = classifier.predict(testNewsArray)

# Show results of the predictor
#for new, category in zip(trainNewsArray, prdct):
#    print('%s => %s' % (new, category))

# Get the accuracy
acc = accuracy_score(testCategoryArray, prdct)
print(acc)

