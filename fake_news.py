# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:27:20 2022

@author: ARIF RAJA MONDAL
"""

import numpy as np
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def fake_news(input_data):
    port_stem = PorterStemmer()
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ',content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    # Downloading Stopwords
    # Obtaining Additional Stopwords From nltk
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    # Removing Stopwords And Remove Words With 2 Or Less Characters
    news_dataset = pd.read_csv('https://github.com/Arif551/Detect_FakeNews_WebApp/blob/main/test.csv')
    # counting the number of missing values in the dataset
    news_dataset.isnull().sum()
    # replacing the null values with empty string
    news_dataset = news_dataset.fillna('')
    # merging the author name and news title
    news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
    # separating the data & label
    X = news_dataset.drop(columns='label', axis=1)
    Y = news_dataset['label']
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    #separating the data and label
    X = news_dataset['content'].values
    Y = news_dataset['label'].values
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    
    X = vectorizer.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    loaded_model = model.fit(X_train, Y_train)



    



   
  
    input_data = stemming(input_data)
    input_data = [input_data]
    
    
    input_data = vectorizer.transform(input_data)
   
    
   
    # Applying The Function To The Dataframe
    
    
    
    
    
    
    
    
    
    
    # Making prediction
    prediction = loaded_model.predict(input_data)
    if(prediction[0]==0):
        return 'The News is Fake'
    else:
        return 'The News is True'
   
    
    
    
def main():
    st.title('Fake news Detection Web app')
    
    title = st.text_input('Type News Title');
    content = st.text_input('Type News Content');
    input_data = title + content
    
    status = ''
    
    if st.button('Detect news Fake or not'):
        status = fake_news(input_data)
        
    st.success(status)
    
    

if __name__ == '__main__':
    main()
