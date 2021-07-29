
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()

# Importing the dataset
dataset = pd.read_csv('Mall_Customers (1).csv')
dataset

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Importing the dataset
dataset = pd.read_csv('/content/Mall_Customers (1).csv')

print(dataset)

#Print Total number of Rows & columns in dataset
print(dataset.shape)

#Print Information about data
dataset.info()

types = dataset.dtypes
print(types)

#Count total number of classes in Data
class_counts = dataset.groupby('Genre').size()
print(class_counts)

from matplotlib import pyplot
dataset.hist()
pyplot.show()

dataset.plot(kind='density' ,subplots=True, layout=(3,3), sharex=False)
pyplot.show()

# Extracting features of dataset

X = dataset.iloc[:, [3, 4]].values

print(X)

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#‘ward’ minimizes the variance of the clusters being merged.
#‘average’ uses the average of the distances of each observation of the two sets.
#‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
#‘single’ uses the minimum of the distances between all observations of the two sets.
dendrogram = sch.dendrogram(sch.linkage(X, method = 'single'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'average'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(X, method='ward'))
plt.axhline(y=200, color='r', linestyle='--')

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

#linkage creates the tree using the specified method, which describes how to measure the distance between clusters
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
#pdist calculates pairwise distance betwween pair of observations
#cophenet computes the cophenetic correlation coefficient for the hierarchical cluster tree represented by Z
#cophenetic distances coph_dists in the same lower triangular distance vector format as X
# c is correlation cofficient
c, coph_dists = cophenet(Z, pdist(X))
c
#coph_dists

A = linkage(X, 'average')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(A, pdist(X))
c

C = linkage(X, 'complete')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(C, pdist(X))
c

Ce = linkage(X, 'centroid')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Ce, pdist(X))
c

S = linkage(X, 'single')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(S, pdist(X))
c

Z[0]

dataset.isnull().sum()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'single')
y_hc = hc.fit_predict(X)

print(y_hc)

# Visualising the clusters
plt.scatter(X[:,0], X[:,1], s = 100, c = 'black', label = 'Data Distribution')
plt.title('Customer Distribution before clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careful')
plt.scatter(X[y_hc== 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

frame = pd.DataFrame(X)
frame['cluster'] = y_hc
frame['cluster'].value_counts()

Annual_Income =  39#@param {type:"number"}
Spending_Score =  91#@param {type:"number"}
Annual_Income1 =  34#@param {type:"number"}
Spending_Score1 =  19#@param {type:"number"}
Annual_Income2 = 34 #@param {type:"number"}
Spending_Score2 =  65#@param {type:"number"}
Annual_Income3 =  45#@param {type:"number"}
Spending_Score3 =  56#@param {type:"number"}
Annual_Income4 =  56#@param {type:"number"}
Spending_Score4 =  87#@param {type:"number"}
predict= hc.fit_predict([[ Annual_Income,Spending_Score ],[ Annual_Income1,Spending_Score1 ], [ Annual_Income2,Spending_Score2 ], [ Annual_Income3,Spending_Score3], [ Annual_Income4,Spending_Score4 ]])
print(predict)

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

print(y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careful')
plt.scatter(X[y_hc== 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

Annual_Income =  39#@param {type:"number"}
Spending_Score =  91#@param {type:"number"}
Annual_Income1 =  34#@param {type:"number"}
Spending_Score1 =  19#@param {type:"number"}
Annual_Income2 = 34 #@param {type:"number"}
Spending_Score2 =  65#@param {type:"number"}
Annual_Income3 =  45#@param {type:"number"}
Spending_Score3 =  56#@param {type:"number"}
Annual_Income4 =  56#@param {type:"number"}
Spending_Score4 =  21#@param {type:"number"}
predict= hc.fit_predict([[ Annual_Income,Spending_Score ],[ Annual_Income1,Spending_Score1 ], [ Annual_Income2,Spending_Score2 ], [ Annual_Income3,Spending_Score3], [ Annual_Income4,Spending_Score4 ]])
print(predict)

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

print(y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careful')
plt.scatter(X[y_hc== 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

Annual_Income =  39#@param {type:"number"}
Spending_Score =  91#@param {type:"number"}
Annual_Income1 =  34#@param {type:"number"}
Spending_Score1 =  19#@param {type:"number"}
Annual_Income2 = 34 #@param {type:"number"}
Spending_Score2 =  65#@param {type:"number"}
Annual_Income3 =  45#@param {type:"number"}
Spending_Score3 =  56#@param {type:"number"}
Annual_Income4 =  56#@param {type:"number"}
Spending_Score4 =  21#@param {type:"number"}
predict= hc.fit_predict([[ Annual_Income,Spending_Score ],[ Annual_Income1,Spending_Score1 ], [ Annual_Income2,Spending_Score2 ], [ Annual_Income3,Spending_Score3], [ Annual_Income4,Spending_Score4 ]])
print(predict)

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage='average')
y_hc = hc.fit_predict(X)

print(y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careful')
plt.scatter(X[y_hc== 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

Annual_Income =  45#@param {type:"number"}
Spending_Score =  45#@param {type:"number"}
Annual_Income1 =  45#@param {type:"number"}
Spending_Score1 =  45#@param {type:"number"}
Annual_Income2 = 45 #@param {type:"number"}
Spending_Score2 =  45#@param {type:"number"}
Annual_Income3 =  45#@param {type:"number"}
Spending_Score3 =  45#@param {type:"number"}
Annual_Income4 =  45#@param {type:"number"}
Spending_Score4 =  45#@param {type:"number"}
Annual_Income5 =  45#@param {type:"number"}
Spending_Score5 =  45#@param {type:"number"}
predict= hc.fit_predict([[ Annual_Income,Spending_Score ],[ Annual_Income1,Spending_Score1 ], [ Annual_Income2,Spending_Score2 ], [ Annual_Income3,Spending_Score3], [ Annual_Income4,Spending_Score4 ], [ Annual_Income5,Spending_Score5 ]])
print(predict)

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(hc) 
  
# Load the pickled model 
Saved_Model = pickle.loads(saved_model)

import joblib
filename = '/content/Mall_Customers (1).sav'
joblib.dump(hc, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(hc,open('/content/Mall_Customers .pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('/content/Mall_Customers .pkl','rb'))  
# Use the loaded pickled model to make predictions

!pip install flask-ngrok

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %mkdir templates -p

# Commented out IPython magic to ensure Python compatibility.
# %%writefile templates/index.html
# <!DOCTYPE html>
# <html >
# <!--From https://codepen.io/frytyler/pen/EGdtg-->
# <head>
# <title>Machine Learning Lab Experiment Deployment</title>
# <meta charset="UTF-8">
# <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
# <link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
# <link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
# <link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
#  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous"> 
# <style><!DOCTYPE html>
# 
# h1 {text-align: center;}
# h2 {text-align: center;}
# h3 {text-align: center;}
# p {text-align: center;}
# div {text-align: center;}
# </style>
# </head>
# 
# <body>
#  
#      
# <div class="" style="background-color:blue;" >
# <div class="clearfix">
#            
# <div class="col-md-12">
# <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
# <center><p style="font-size:30px;color:white;margin-top:10px;">Computer Engineering</p></center> 
# <center><p style="font-size:25px;color:white;margin-top:10px;">Machine Learning Experiment </p></center> 
# </div>
# </div>
# </div>
# 
# <div class="login">
# <h2 >K Means clustering Customer segmentation</h2>
# <h4>Developed by Chandan</h4>
# <!-- Main Input For Receiving Query to our ML -->
# <form action="{{ url_for('predict')}}"method="get">
# 
#  
#     <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">Annual Income in $ for First customer</label>
#       <input type="number" name="income1" id="income1" value="" min="1" max="140" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score for First customer</label>
#       <input type="number"  name="score1" id="score1" value=""min="1" max="100" placeholder="1" required="required">
#       </div>
#   <label for="exampleFormControlInput1" class="form-label">Annual Income in for second customer $</label>
#       <input type="number" name="income2" id="income2" value="" min="1" max="140" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score for second customer</label>
#       <input type="number"  name="score2" id="score2" value=""min="1" max="100" placeholder="1" required="required">
#       </div>
#        <div class="mb-3">
#       <label for="exampleFormControlInput1" class="form-label">Annual Income in for Third Customer in  $</label>
#       <input type="number" name="income3" id="income3" value="" min="1" max="140" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score for Third customer</label>
#       <input type="number"  name="score3" id="score3" value=""min="1" max="100" placeholder="1" required="required">
#       </div>
#        <div class="mb-3">
#       <label for="exampleFormControlInput1" class="form-label">Annual Income in for Forth Customer $</label>
#       <input type="number" name="income4" id="income4" value="" min="1" max="140" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score for Fourth customer</label>
#       <input type="number"  name="score4" id="score4" value=""min="1" max="100" placeholder="1" required="required">
#       </div>
#        <div class="mb-3">
#       <label for="exampleFormControlInput1" class="form-label">Annual Income in for Fifth Customer $</label>
#       <input type="number" name="income5" id="income5" value="" min="1" max="140" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score for Fifth customer</label>
#       <input type="number"  name="score5" id="score5" value=""min="1" max="100" placeholder="1" required="required">
#       </div>
#   <div class="col-auto">
#  
#     <button type="submit" class="btn btn-danger">predict type of Customer</button>
#   </div>
# </form>
# 
# <br>
# <br>
# {{ prediction_text }}
# 
# </div>
# 
# <div class="" style="background-color:blue;" >
# <div class="clearfix">
#            
# <div class="col-md-12">
#  <center><p style="font-size:25px;color:white;margin-top:20px;">Machine Learning Experiment </p></center> 
# </div>
# </div>
# </div>
# </body>
# </html>

!pip install flask-ngrok
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('/content/Mall_Customers .pkl','rb'))     
run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  income1 = int(request.args.get('income1'))
  score1 = int(request.args.get('score1'))
  income2 = int(request.args.get('income2'))
  score2 = int(request.args.get('score2'))  
  income3 = int(request.args.get('income3'))
  score3 = int(request.args.get('score3')) 
  income4 = int(request.args.get('income4'))
  score4 = int(request.args.get('score4')) 
  income5 = int(request.args.get('income5'))
  score5 = int(request.args.get('score5'))      
  predict = model.fit_predict([[income1,score1 ],[income2,score2], [income3,score3],[income4,score4], [income5,score5]])
  
        
  return render_template('index.html', prediction_text='Model  has predicted  : {}'.format(predict))


app.run()

