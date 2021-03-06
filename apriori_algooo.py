

# Importing the libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv('retail_dataset.csv')
dataset.head()

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

print(dataset)

#Print Total number of Rows & columns in dataset
print(dataset.shape)

#Print Information about data
dataset.info()

types = dataset.dtypes
print(types)

#Count total number of classes in Data
items = (dataset['0'].unique())
items

dataset.isnull().sum()

#Create list 
transactions = []
for i in range(0, 315):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

transactions

"""The apriori function expects data in a one-hot encoded pandas DataFrame. We can transform it into the right format via the TransactionEncoder as follows:"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]

print(df)

freq_items = apriori(df, min_support=0.2, use_colnames=True)
freq_items

rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
rules

list(rules)

print(len(rules))

freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
freq_items

freq_items[ (freq_items['length'] == 2) &
                   (freq_items['support'] >= 0.3) ]

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(freq_items,open('/content/drive/MyDrive/Machine learning/retail_dataset.pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('/content/drive/MyDrive/Machine learning/retail_dataset.pkl','rb'))  
# Use the loaded pickled model to make predictions

!pip install streamlit

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install pyngrok

!ngrok authtoken 1oEm0wopEJyjrT38ULluwUKK5fq_7ai4ZocZJ2YuFuoiJfoMh

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st 
# from PIL import Image
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules
# st.set_option('deprecation.showfileUploaderEncoding', False)
# def find_associatio_rule(support):
#   # Load the pickled model
#   model = pickle.load(open('/content/drive/MyDrive/Machine learning/retail_dataset.pkl','rb'))     
#   if uploaded_file is not None:
#     dataset= pd.read_csv(uploaded_file)
#   else:
#     dataset= pd.read_csv('/content/drive/MyDrive/Machine learning/retail_dataset.csv')
# 
#   #Create list 
#   transactions = []
#   for i in range(0, 315):
#     transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])
# 
#   from mlxtend.preprocessing import TransactionEncoder
#   te = TransactionEncoder()
#   te_ary = te.fit(transactions).transform(transactions)
#   df = pd.DataFrame(te_ary, columns=te.columns_)
#   df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
#   freq_items = apriori(df, min_support=support, use_colnames=True)
#   rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
#   return rules
# def find_frequent_items(support):
#   # Load the pickled model
#   model = pickle.load(open('/content/drive/MyDrive/Machine learning/retail_dataset.pkl','rb'))     
#   if uploaded_file is not None:
#     dataset= pd.read_csv(uploaded_file)
#   else:
#     dataset= pd.read_csv('/content/drive/MyDrive/Machine learning/retail_dataset.csv')
#   #Create list 
#   transactions = []
#   for i in range(0, 315):
#     transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])
# 
#   from mlxtend.preprocessing import TransactionEncoder
#   te = TransactionEncoder()
#   te_ary = te.fit(transactions).transform(transactions)
#   df = pd.DataFrame(te_ary, columns=te.columns_)
#   df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
#   freq_items = apriori(df, min_support=support, use_colnames=True)
#   rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
#   
#   return freq_items
# html_temp = """
#    <div class="" style="background-color:blue;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
#    <center><p style="font-size:30px;color:white;margin-top:10px;">Computer Engineering</p></center> 
#    <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning</p></center> 
#    </div>
#    </div>
#    </div>
#    """
# st.markdown(html_temp,unsafe_allow_html=True)
# st.header("Identification of items Purchased together ")
#   
# uploaded_file = st.file_uploader("Upload dataset", help='Please upload retail_dataset.csv otherwise leave  blank') 
# support = st.number_input('Insert a minimum suppport to find association rule ',0,1)
# 
#   
# if st.button("Association Rule"):
#   rules=find_associatio_rule(support)
#   st.success('Apriori has found Following rules {}'.format(rules))
# if st.button("Frequent Items"):
#   frequent_items=find_frequent_items(support)
#   st.success('Apriori has found Frequent itemsets {}'.format(frequent_items))      
# if st.button("About"):
#   st.subheader("Developed by Chandan Kumar")
#   st.subheader("Computer Engineering")
# html_temp = """
#    <div class="" style="background-color:orange;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine learning</p></center> 
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

!pip install apyori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('retail_dataset.csv', sep=',', header = None )
dataset.head()

print(dataset)

transactions = []
for i in range(0, 315):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

transactions

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.2, min_confidence = 0.3, min_lift = 1, min_length = 1)

# Visualising the results
results = list(rules)
print(results)

for  rule in results:
  for Rlationrecors in rule:
    print(Rlationrecors)

