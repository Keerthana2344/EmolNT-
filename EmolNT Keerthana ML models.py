#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Now I have to Load data from text file
data_url = "https://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data/anger-ratings-0to1.dev.target.txt"
data = pd.read_csv(data_url, header=None, delimiter='\t', names=['Text', 'Intensity'])

# Preprocessing data
X = data['Text']
y = data['Intensity']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Ridge Regression": Ridge(),
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(),
    "Random Forest Regressor": RandomForestRegressor()
}

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform text data to TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"{name} RMSE: {rmse:.4f}")


# In[2]:





# In[3]:





# In[ ]:




