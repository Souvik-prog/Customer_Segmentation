# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import the dataset
df = pd.read_excel('OnlineRetail.xlsx')
df = df[df['CustomerID'].notna()]

# Sample the dataset
df_fix = df.sample(10000, random_state = 42)
#print(df_fix.head())

# Recency , Frequency and Monetary Value (RFM) table
df_fix['TotalSum'] = df_fix['UnitPrice']*df_fix['Quantity']
df_fix['InvoiceDate'] = df_fix['InvoiceDate'].dt.date
snapshot_date = max(df_fix['InvoiceDate']) + datetime.timedelta(days=1)

#Aggregating Customers
customer = df_fix.groupby(['CustomerID']).agg({
    'TotalSum' : 'sum',
    'InvoiceNo' : 'count',
    'InvoiceDate' : lambda x : (snapshot_date - x.max()).days
})

# Rename the columns
customer.rename(columns={
    'InvoiceNo' : 'Frequency',
    'InvoiceDate' : 'Recency',
    'TotalSum' : 'MonetaryValue'
}, inplace = True)

#print(customer.head())
#sns.displot(customer['MonetaryValue'])
#plt.show()
# Process the data

# Boxcox transformation on the data
customer_fix = pd.DataFrame()
customer_fix['Recency']  = stats.boxcox(customer['Recency'])[0]
customer_fix['Frequency'] = stats.boxcox(customer['Frequency'])[0]
customer_fix['MonetaryValue'] = pd.Series(np.cbrt(customer['MonetaryValue'])).values
#print(customer_fix.tail())
#sns.displot(customer_fix['Frequency'])
#plt.show()

# Normalising the data i.e. each variable has mean 0 and variance 1
scaler = StandardScaler()
scaler.fit(customer_fix)
customer_normalized = scaler.transform(customer_fix)
#print(customer_normalized)
# Mean of Normalised data
# print(customer_normalized.mean(axis=0).round(2))
# Standard Deviation of Normalised data
# print(customer_normalized.std(axis=0).round(2))

# Model the data
# Elbow Method
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_normalized)
    sse[k] = kmeans.inertia_ # Calculates how good a model is i.e. Sum of Squares Error

#  A good model has low SSE and lower Number of Clusters

#plt.title('Elbow Method')
#plt.xlabel('Number of Clusters')
#plt.ylabel('SSE')
#sns.pointplot(x=list(sse.keys()), y=list(sse.values()))

# From the plot we can infer , optimal value of k is 3
#plt.show()

model = KMeans(n_clusters=3, random_state=42)
model.fit(customer_normalized)
# print(model.labels_.shape)

# Perform KMeans on the RFM table and calculate the mean of each variable
customer['Cluster'] = model.labels_
customer_mean  = customer.groupby('Cluster').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'MonetaryValue' : ['mean', 'count']
}).round(2)
#print(customer_mean)

# Create the normalized dataframe
df_normalized = pd.DataFrame(customer_normalized, columns=
                             ['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customer.index
df_normalized['Cluster'] = model.labels_

# print(df_normalized.head())
df_melt = pd.melt(df_normalized.reset_index(),
                  id_vars=['ID', 'Cluster'],
                  value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                  var_name='Attribute',
                  value_name='Value')
#print(df_melt.head())
sns.lineplot('Attribute', 'Value', hue='Cluster', data = df_melt)
plt.show()

