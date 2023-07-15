
!pip install -q kaggle

from google.colab import files
files.upload()

! mkdir ~/.kaggle #create kaggle folder
! cp kaggle.json ~/.kaggle/ #copy kaggle json to the folder

! chmod 600 ~/ .kaggle/kaggle.json #permission for json to act
! kaggle datasets list #list all dataset in kaggle

! kaggle datasets download -d zarajamshaid/language-identification-datasst

! unzip language-identification-datasst.zip

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer#Text to Numerical Data
from sklearn.model_selection import train_test_split #Slip raation of train and Test data sets
from sklearn.naive_bayes import MultinomialNB #
le = LabelEncoder() # Catgorical to numerical Data
data = pd.read_csv("/content/dataset.csv")
print(data.head())

data.isnull().sum()
data["language"].value_counts()

X= data["Text"]
y = data["language"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
for text in X:
  text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
  text = re.sub(r'[[]]', ' ', text)
  text = text.lower()
  data_list.append(text)
tfidf = TfidfVectorizer()

from sklearn.model_selection import train_test_split
X = tfidf.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y,  test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)

def predict(text):
     x = tfidf.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language

predict("나는 엔지니어입니다")

predict("how is the weather?")

predict('Je suis ingénieur')

predict('आप सुंदर हैं')

predict('நீ அழகாக இருக்கிறாய்')
