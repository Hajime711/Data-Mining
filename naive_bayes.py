import numpy as np 
import pandas as pd 

data = pd.read_csv("E:\Progam\Python\spam.csv")
print(data.head())

from sklearn.feature_extraction.text import CountVectorizer
CV=CountVectorizer()
xtrain=CV.fit_transform(data.Message)
xtest=CV.transform(data.Category)
xtrain.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
xtrain_tfidf=tfidf.fit_transform(xtrain)
xtest_tfidf=tfidf.transform(xtest)
xtrain_tfidf.shape
df_idf=pd.DataFrame(tfidf.idf_,index=CV.get_feature_names_out(),columns=["IDF_WEIGHTS"])
df_idf.sort_values(by=["IDF_WEIGHTS"]).head(10)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()                           
model.fit(xtrain_tfidf,data.Category)
ypred=model.predict(xtest_tfidf)
ypred

from sklearn.metrics import accuracy_score
print("Accuracy Score : " , accuracy_score(ypred,data.Category)*100)


