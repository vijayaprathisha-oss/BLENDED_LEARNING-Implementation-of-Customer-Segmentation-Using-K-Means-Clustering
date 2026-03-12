# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, seaborn, matplotlib, and sklearn.
2. Load the dataset CustomerData.csv and Select important features (Age, Annual Income, Spending Score).
3. Apply StandardScaler to normalize the feature values.
4. Use the Elbow Method to determine the optimal number of clusters.
5. Train the K-Means clustering model with the optimal number of clusters.
6. Assign cluster labels to each data point.
7. Calculate the Silhouette Score to evaluate clustering performance.
8. Visualize the clusters using a scatter plot.
   
## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
data=pd.read_csv('CustomerData.csv')
print(data.head())
print(data.columns)
features=['Age','Annual Income (k$)','Spending Score (1-100)']
X=data[features]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,4))
plt.plot(range(1,11),wcss,marker='o',linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters=4
kmeans=KMeans(n_clusters=optimal_clusters,random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("Name: VIJAYAPRATHISHA J")
print("Ref No:212225240184")
print(f'Silhouette Score: {sil_score}')

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

## Output:
<img width="770" height="208" alt="image" src="https://github.com/user-attachments/assets/06c50a2a-e8c5-4728-aaf6-c8ac54bf52ae" />

<img width="913" height="508" alt="image" src="https://github.com/user-attachments/assets/780a21a3-b730-4f8c-bd3b-16fba3a9407b" />

<img width="1154" height="772" alt="image" src="https://github.com/user-attachments/assets/4a80dd0f-b2e8-4219-abcb-6ba07286a2a5" />

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
