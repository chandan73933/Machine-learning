
# Data Preprocessing

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

dataset.describe()

# Extracting features of dataset

X = dataset.iloc[:, [3, 4]].values

print(X)

dataset.isnull().sum()

"""kmeans++ 1). Randomly select the first centroid from the data points. 2). For each data point compute its distance from the nearest, previously choosen centroid. 3). Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid) 4)Repeat steps 2 and 3 untill k centroids have been sampled"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
print(kmeans)
y_kmeans = kmeans.fit_predict(X)



"""kmeans++ 1). Randomly select the first centroid from the data points. 2). For each data point compute its distance from the nearest, previously choosen centroid. 3). Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid) 4)Repeat steps 2 and 3 untill k centroids have been sampled"""

print("Within cluster sum of square when k=5", kmeans.inertia_)

print("center of Cluster are", kmeans.cluster_centers_ )

print("Number of iterations", kmeans.n_iter_)

print(X[:,0])

# Visualising the clusters
plt.scatter(X[:,0], X[:,1], s = 100, c = 'black', label = 'Data Distribution')
plt.title('Customer Distribution before clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

frame = pd.DataFrame(X)
frame['cluster'] = y_kmeans
frame['cluster'].value_counts()

Annual_Income =  56#@param {type:"number"}
Spending_Score = 92 #@param {type:"number"}

predict= kmeans.predict([[ Annual_Income,Spending_Score ]])
print(predict)
if predict==[0]:
  print("Customer is careless")

elif predict==[1]:
  print("Customer is standard")
elif predict==[2]:
  print("Customer is Target")
elif predict==[3]:
  print("Customer is careful")

else:
  print("Custmor is sensible" )

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans== 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careless')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careful')
plt.scatter(X[y_kmeans== 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(kmeans) 
  
# Load the pickled model 
Saved_Model = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
Saved_Model.predict(X)

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

import joblib
filename = '/content/Mall_Customers .sav'
joblib.dump(kmeans, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(kmeans,open('/content/kmeanscluster.pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('/content/kmeanscluster.pkl','rb'))  
# Use the loaded pickled model to make predictions 
model.predict(X)

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
# <div class="" style="background-color:salmon;" >
# <div class="clearfix">
#            
# <div class="col-md-12">
# <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
# <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
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
#   <label for="exampleFormControlInput1" class="form-label">Annual Income in $(15-137)</label>
#       <input type="number" name="income" id="income" value="" min="15" max="137" placeholder="" required="required">
#       </div>
#       <div class="mb-3">
#   <label for="exampleFormControlInput1" class="form-label">spending Score(1-99)</label>
#       <input type="number"  name="score" id="score" value=""min="1" max="99" placeholder="1" required="required">
#       </div>
#   
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
# <div class="" style="background-color:salmon;" >
# <div class="clearfix">
#            
# <div class="col-md-12">
#  <center><p style="font-size:25px;color:white;margin-top:20px;">Machine Learning Experiment </p></center>
#  <center><p style="font-size:25px;color:white;margin-top:20px;">Developed by Chandan</p></center> 
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
model = pickle.load(open('/content/kmeanscluster.pkl','rb'))   
run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  income = int(request.args.get('income'))
  score = int(request.args.get('score'))
    
    
  predict = model.predict([[income,score ]])
  if predict==[0]:
    result="Customer is careless"

  elif predict==[1]:
    result="Customer is standard"
  elif predict==[2]:
    result="Customer is Target"
  elif predict==[3]:
    result="Customer is careful"

  else:
    result="Custmor is sensible"
    
        
  return render_template('index.html', prediction_text='Model  has predicted  : {}'.format(result))


app.run()

