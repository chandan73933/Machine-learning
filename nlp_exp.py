
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

#  read the CSV file and look at the first five rows of the data:
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t", quoting=3)
dataset.head()

print(dataset)

#Print Total number of Rows & columns in dataset
print(dataset.shape)

#Print Information about data
dataset.info()

#Data types of Columns
types = dataset.dtypes
print(types)

#Count total number of classes in Data
class_counts = dataset.groupby('Liked').size()
print(class_counts)

#Count total number of classes in Data
class_counts = dataset.groupby('Review').size()
print(class_counts)

# Histogram plot
from matplotlib import pyplot
dataset.hist()
pyplot.show()

#Density Plot - representation of distribution of numerical values 
dataset.plot(kind='density' ,subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#Finding missing values
dataset.isnull().sum()

# Let us take text to understand process of data preprocessing in NLP
text="Wow... Loved this place."

print(dataset['Review'][0])

# First step: cleaning Text and removing number and punctuation marks.
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
print(text)
print('-------------------------------------------------')
print('Review after removing number and punctuation marks. ')
print(review)

# Second  step: converting text into lower case.
review=review.lower()
print(text)
print('-------------------------------------------------')
print('Text after convering into lower case')
print(review)

# Third step: Removing stop words like 'this, the'
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review = review.split()
print(review)

# Third step: Removing stop words like 'this, the'
# set function is generally used for long article to fastem process
review1 = [word for word in review if not word in set(stopwords.words('english'))]
print('Text after removing stop words')
print(review1)

# Fourth step: converting stemming words
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
print('After converting stemmer words')
print(review)

# joining these words of list
review2 = ' '.join(review)

print(review2)

# Creating the Bag of Words model
corpus1 = []
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(review2)
corpus1.append(review2)
print(corpus1)
X = cv.fit_transform(corpus1)
print(X)

# Cleaning the texts for all review using for loop
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    print(review)
    corpus.append(review)

print(corpus)

#    Adding corpus to csv 
corpus_dataset = pd.DataFrame(corpus)
corpus_dataset['corpus'] = corpus_dataset
corpus_dataset = corpus_dataset.drop([0], axis = 1) 
corpus_dataset.to_csv('/content/drive/MyDrive/Machine learning/Restaurant_Reviews.csv')

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

print(X)
print(X.shape)

print(y)
print(y.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
print(cm)

# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)*100))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))

# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(classifier) 
  
# Load the pickled model 
model = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
model.predict(X_test)

# save the model to disk
import joblib
filename = '/content/drive/MyDrive/Machine learning/Restaurant_Reviews_nlp.sav'
joblib.dump(classifier, filename)
 
# some time later...
 
# load the model from disk
naive_bayes_model = joblib.load(filename)
result = naive_bayes_model.score(X_test, y_test)
print(result)

Review = "very good" #@param {type:"string"}
input_data = [Review] 
  
input_data = cv.transform(input_data).toarray()
print(input_data)

input_pred = classifier.predict(input_data)

input_pred = input_pred.astype(int)


if input_pred[0]==1:
    print("Review is Positive")
else:
    print("Review is Negative")

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(classifier,open('/content/drive/MyDrive/Machine learning/Restaurant_Reviews_nlp.pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('/content/drive/MyDrive/Machine learning/Restaurant_Reviews_nlp.pkl','rb'))  
# Use the loaded pickled model to make predictions

!pip install streamlit

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install pyngrok

#!ngrok authtoken 1oEm0wopEJyjrT38ULluwUKK5fq_7ai4ZocZJ2YuFuoiJfoMh
!ngrok authtoken 1rbzRlIU1f9PwpoYMgJ5ju8yE5a_5An5x3qSosntavQiDXzsc

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st 
# from PIL import Image
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# st.set_option('deprecation.showfileUploaderEncoding', False)
# # Load the pickled model
# model = pickle.load(open('/content/drive/MyDrive/Machine learning/Restaurant_Reviews.pkl','rb'))   
# 
# 
# def review(text):
#   dataset = pd.read_csv('/content/drive/MyDrive/Machine learning/Restaurant_Reviews.tsv', delimiter="\t", quoting=3)
#   # First step: cleaning Text and removing number and punctuation marks.
#   # Cleaning the texts for all review using for loop
#   import re
#   import nltk
#   nltk.download('stopwords')
#   from nltk.corpus import stopwords
#   from nltk.stem.porter import PorterStemmer
#   corpus = []
#   for i in range(0, 1000):
#     review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     #print(review)
#     corpus.append(review)
#   # Creating the Bag of Words model
#   from sklearn.feature_extraction.text import CountVectorizer
#   cv = CountVectorizer(max_features = 1500)
#   #print(cv)
#   X = cv.fit_transform(corpus).toarray()
#   import re
#   review = re.sub('[^a-zA-Z]', ' ', text)
#   review=review.lower()
#   print(review)
#   # Third step: Removing stop words like 'this, the'
#   import nltk
#   nltk.download('stopwords')
#   from nltk.corpus import stopwords
#   review = review.split()
#   print(review)
#   # Third step: Removing stop words like 'this, the'
#    # set function is generally used for long article to fastem process
#   review1 = [word for word in review if not word in set(stopwords.words('english'))]
#   print(review1)
#   # Fourth step: converting stemming words
#   from nltk.stem.porter import PorterStemmer
#   ps = PorterStemmer()
#   review = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
#   print(review)
#   # joining these words of list
#   review2 = ' '.join(review)
#   print(review2)
#   # Creating the Bag of Words model
#   
#   X = cv.transform(review).toarray()
#   input_pred = model.predict(X)
#   input_pred = input_pred.astype(int)
#   print(input_pred)
#   if input_pred[0]==1:
#     result= "Review is Positive"
#   else:
#     result="Review is negative" 
# 
#  
#     
#   return result
# html_temp = """
#    <div class="" style="background-color:blue;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
#    <center><p style="font-size:30px;color:white;margin-top:10px;">Computer Engineering</p></center> 
#    <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning </p></center> 
#    </div>
#    </div>
#    </div>
#    """
# st.markdown(html_temp,unsafe_allow_html=True)
# st.header("Restaurant Review System ")
#   
#   
# text = st.text_area("Writre Review of Restaurant")
# 
# if st.button("Review Analysis"):
#   result=review(text)
#   st.success('Model has predicted {}'.format(result))
#       
# if st.button("About"):
#   st.subheader("Developed by Chandan")
#   
# html_temp = """
#    <div class="" style="background-color:orange;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine learning NLP</p></center> 
#    </div>
#    </div>
#    </div>
#    """
# st.markdown(html_temp,unsafe_allow_html=True)

!nohup streamlit run  app.py &

from pyngrok import ngrok
url=ngrok.connect(port='8051')
url

!streamlit run --server.port 80 app.py

# Random Forest

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.3f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.3f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % (score*100))

# Fitting Decision tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

# Fitting K Nearest Neighbor classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

# Fitting Logistic classifier to the Training set
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

# Fitting Logistic classifier to the Training set
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ')
#print(cm)
# calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# calculate precision
# Precision = TruePositives / (TruePositives + FalsePositives)
precision = precision_score(y_test, y_pred, average='binary')
print('Precision: %.2f' % (precision*100))
# calculate recall
# Recall = TruePositives / (TruePositives + FalseNegatives)
recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.2f' % (recall*100))
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
# calculate score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.2f' % (score*100))

