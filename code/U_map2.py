#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:33:32 2024

@author: ali
"""

import os
import re
import pandas as pd
import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_lsq_spline
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib import cm
from scipy.stats import percentileofscore
import seaborn as sns
from scipy.stats import linregress, t, stats
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from statannotations.Annotator import Annotator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from metric_learn import NCA

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np


# Function to z-score a single matrix
def z_score_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)
    z_scored_matrix = (matrix - mean) / std_dev
    return matrix
    #return z_scored_matrix
    
def threshold_matrix(matrix, percentile=30):
    # Flatten the matrix to sort and find the threshold value
    flat_matrix = matrix.flatten()
    threshold_value = np.percentile(flat_matrix, percentile)  # Get the 20th percentile value
    # Apply threshold
    matrix[matrix < threshold_value] = 0
    #print(threshold_value)
    return matrix
    
# Directory containing the CSV files
directory = '/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/connectome/DTI/plain/'

subj_list = []
matrices = []
matrix_size = (84,84)

# Iterate through files in the directory
for filename in os.listdir(directory):
    if re.search("\d_conn_plain.csv$",filename):
        file_path = os.path.join(directory, filename)
        # Read CSV file into a dataframe
        subj_df = pd.read_csv(file_path, header=None)  # Assuming there is no header
        if subj_df.shape == matrix_size:
            subj_list.append(int(os.path.basename(file_path)[4:8]))
            z_scored_matrix = z_score_matrix(subj_df.to_numpy())
            thresholded_matrix = threshold_matrix(z_scored_matrix, 10)
            matrices.append(thresholded_matrix)
df_conn_plain = pd.DataFrame(index=subj_list)
df_conn_plain['matrices']  = matrices
    
dir_fmri = '/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/connectome/fMRI'

subj_list = []
matrices = []
matrix_size = (84,84)

# Iterate through files in the directory
for filename in os.listdir(dir_fmri):
    if re.search("func_connectome_corr_",filename):
        file_path = os.path.join(dir_fmri, filename)
        # Read CSV file into a dataframe
        subj_df = pd.read_csv(file_path, header=None)  # Assuming there is no header
        if subj_df.shape == matrix_size:
            subj_list.append(int(os.path.basename(file_path)[25:29]))
            #z_scored_matrix = z_score_matrix(subj_df.to_numpy())
            #thresholded_matrix = threshold_matrix(z_scored_matrix, 10)
            matrices.append(subj_df.to_numpy())
df_conn_fmri = pd.DataFrame(index=subj_list)
df_conn_fmri['matrices']  = matrices


df_conn_plain = df_conn_plain.reset_index().rename(columns={'index': 'id'})
df_conn_fmri = df_conn_fmri.reset_index().rename(columns={'index': 'id'})
    

merged_df = pd.merge(df_conn_plain, df_conn_fmri, on='id', suffixes=('_plain', '_fmri'))






df_metadata = pd.read_excel("/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/metadata/alex-badea_2024-06-14.xlsx")
##### drop ones without biomarkers
df_metadata = df_metadata.dropna(subset=['PTAU181'])
len(df_metadata)
### aslo ratio of ab
df_metadata ['ABratio']  =  (df_metadata['AB42'] / df_metadata['AB40'] )
df_metadata ['ABratio']  =  df_metadata ['ABratio'] / np.max(df_metadata ['ABratio'])
df_metadata ['PTAU181']  =  df_metadata ['PTAU181'] / np.max(df_metadata ['PTAU181'])
df_metadata ['NFL']  =  df_metadata ['NFL'] / np.max(df_metadata ['NFL'])




df_metadata['PTID'] =  df_metadata['PTID'].str.slice(start=4 , stop = 8 ).apply(int)


df_metadata.set_index('PTID', inplace=True)

df = merged_df.join(df_metadata, how='inner')

##### learning distance parameter

df['DEMENTED'].fillna(0, inplace=True)

X = np.array( [matrix.flatten() for matrix in df.matrices_plain] )
y = np.array( df.DEMENTED)



nca = NCA(random_state=2)
nca.fit(X, y)

metric_fun = nca.get_metric()

metric_fun( X[1],X[2])
###############


X = np.array( [matrix.flatten() for matrix in df.matrices_fmri] )
y = np.array( df.DEMENTED)



nca = NCA(random_state=2)
nca.fit(X, y)

metric_fun3 = nca.get_metric()

metric_fun( X[1],X[2])






#########
X = np.array(df[['ABratio' , 'PTAU181', 'NFL'] ]  )
y = np.array( df.DEMENTED)



nca2 = NCA(random_state=2)
nca2.fit(X, y)

metric_fun2 = nca2.get_metric()

#metric_fun( X[1],X[2])










# Function to calculate Wasserstein distance between two matrices
def calculate_distance(matrix1, matrix2):
    return wasserstein_distance(matrix1.flatten(), matrix2.flatten())

def matrix_of_distances(matrices_a, matrices_b):
    distances = np.zeros((len(matrices_a), len(matrices_b)))
    for i, matrix_a in enumerate(matrices_a):
        for j, matrix_b in enumerate(matrices_b):
            distances[i, j] = metric_fun(matrix_a.flatten(), matrix_b.flatten())
    return distances


def matrix_of_distances_fmri(matrices_a, matrices_b):
    distances = np.zeros((len(matrices_a), len(matrices_b)))
    for i, matrix_a in enumerate(matrices_a):
        for j, matrix_b in enumerate(matrices_b):
            distances[i, j] = metric_fun3(matrix_a.flatten(), matrix_b.flatten())
    return distances

def biom_of_distances(df1, df2):
    distance_matrix = np.zeros((df1.shape[0], df2.shape[0]))
    for i in range(df1.shape[0]):
        for j in range(df2.shape[0]):
            distance_matrix[i, j] = metric_fun2(df1.iloc[i], df2.iloc[j])
    return distance_matrix

# Filter rows where 'risk_for_ad' is 2 or 3
risky = df[df['DEMENTED'].isin([1])]
print(risky)




df_distances = pd.DataFrame(matrix_of_distances(df['matrices_plain'],risky['matrices_plain']),index=df.index, columns=risky.index)

#fmri 

df_distances_fmri = pd.DataFrame(matrix_of_distances_fmri(df['matrices_fmri'],risky['matrices_fmri']),index=df.index, columns=risky.index)


##### bio distance


df_distances_bio = pd.DataFrame(biom_of_distances(df[['ABratio' , 'PTAU181', 'NFL'] ], risky[['ABratio' , 'PTAU181', 'NFL'] ]),index=df.index, columns=risky.index)


df_distances['average_distance'] = df_distances.mean(axis=1)
df = df.join(df_distances['average_distance'])


df_distances_fmri['average_distance_fmri'] = df_distances_fmri.mean(axis=1)
df = df.join(df_distances_fmri['average_distance_fmri'])




df_distances_bio['average_distance_bio'] = df_distances_bio.mean(axis=1)
df = df.join(df_distances_bio['average_distance_bio'])


filtered = df 


#simmilarity_both= (1/filtered['average_distance']) / np.max(1/filtered['average_distance']) + (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
#simmilarity_both = simmilarity_both /2

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))





simmilarity_bio=  (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio']) 
model = LinearRegression()
age_screen  = df[['SUBJECT_AGE_SCREEN']]
model.fit(df[['SUBJECT_AGE_SCREEN']].values, simmilarity_bio)
# Predict the values of simmilarity_bio based on SUBJECT_AGE_SCREEN
predicted_values = model.predict(age_screen)
# Regress out SUBJECT_AGE_SCREEN by subtracting the predicted values
residuals = simmilarity_bio - predicted_values
residuals = residuals.to_numpy()
simmilarity_bio   = scaler.fit_transform (residuals.reshape(-1, 1) )


simmilarity_conn = (1/filtered.average_distance)/max(1/filtered.average_distance)
model.fit(df[['SUBJECT_AGE_SCREEN']].values, simmilarity_conn)
predicted_values = model.predict(age_screen)
residuals = simmilarity_conn - predicted_values
residuals = residuals.to_numpy()
simmilarity_conn   = scaler.fit_transform (residuals.reshape(-1, 1) )




simmilarity_conn_fmri = (1/filtered.average_distance_fmri)/max(1/filtered.average_distance_fmri)
model.fit(df[['SUBJECT_AGE_SCREEN']].values, simmilarity_conn_fmri)
predicted_values = model.predict(age_screen)
residuals = simmilarity_conn_fmri - predicted_values
residuals = residuals.to_numpy()
simmilarity_conn_fmri   = scaler.fit_transform (residuals.reshape(-1, 1) )






similarity_dt_fm =   (simmilarity_conn  + simmilarity_conn_fmri )/2
similarity_dt_bio =  (simmilarity_conn  + simmilarity_bio )/2
similarity_bio_fm =   (simmilarity_bio  + simmilarity_conn_fmri )/2
similarity_all =   (simmilarity_bio  + simmilarity_conn_fmri + simmilarity_conn )/3


filtered['simmilarity_bio'] = simmilarity_bio
filtered['simmilarity_conn'] = simmilarity_conn
filtered['simmilarity_conn_fmri'] = simmilarity_conn_fmri
filtered['similarity_dt_fm'] = similarity_dt_fm
filtered['similarity_dt_bio'] = similarity_dt_bio
filtered['similarity_bio_fm'] = similarity_bio_fm
filtered['similarity_all'] = similarity_all








import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table



# Add a constant term for the intercept
filtered['intercept'] = 1.0


# Define the independent variables (predictors)
X = filtered[['intercept', 'simmilarity_bio', 'simmilarity_conn', 'simmilarity_conn_fmri']]

# Define the dependent variable (response)
y = filtered['DEMENTED']

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Print the summary of the model
print(result.summary())


# Get the estimated probabilities (likelihood) for each subject
filtered['personalized_risk'] = result.predict(X)

# Display the personalized risk for each subject
print(filtered[['personalized_risk']])



# Perform Wald test for each coefficient
wald_test = result.wald_test_terms()
print(wald_test)








import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap



filtered['APOE_num'] = filtered['APOE'].map({'2/4': 1, '2/3':0, '3/3': 0, '3/4': 1, '4/4' : 1})
filtered['SUBJECT_AGE_SCREEN_z'] = (filtered['SUBJECT_AGE_SCREEN'] - np.mean(filtered['SUBJECT_AGE_SCREEN']) )/ np.std(filtered['SUBJECT_AGE_SCREEN'])
filtered['SUBJECT_SEX_01'] = filtered['SUBJECT_SEX']-1

filtered['BMI_z'] = (filtered['BMI'] - np.mean(filtered['BMI']) )/ np.std(filtered['BMI'])
filtered['BPSYS_AVG_z'] = (filtered['BPSYS_AVG'] - np.mean(filtered['BPSYS_AVG']) )/ np.std(filtered['BPSYS_AVG'])
filtered['BPDIA_AVG_z'] = (filtered['BPDIA_AVG'] - np.mean(filtered['BPDIA_AVG']) )/ np.std(filtered['BPDIA_AVG'])




# Concatenate the features along the second axis (columns)
features = filtered [['SUBJECT_AGE_SCREEN_z', 'SUBJECT_SEX_01', 'APOE_num', 'simmilarity_bio' , 'simmilarity_conn', 'simmilarity_conn_fmri' ]]


# Apply UMAP to reduce dimensionality
reducer = umap.UMAP(n_components=3)
embedding = reducer.fit_transform(features)

# Apply K-means clustering to the reduced data
kmeans = KMeans(n_clusters=5, n_init=20)
clusters = kmeans.fit_predict(embedding)



cmap = ListedColormap(['red', 'green', 'blue', 'orange', 'purple'])





# Visualize the UMAP embedding with cluster coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap=cmap , s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('UMAP projection of the data (colored by clusters)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()


# Visualize the UMAP embedding with cluster coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 1], embedding[:, 2], c=clusters, cmap=cmap, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('UMAP projection of the data (colored by clusters)')
plt.xlabel('UMAP2')
plt.ylabel('UMAP3')
plt.show()




# Visualize the UMAP embedding with cluster coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 2], c=clusters, cmap=cmap, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('UMAP projection of the data (colored by clusters)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP3')
plt.show()



features = filtered [['SUBJECT_AGE_SCREEN_z', 'SUBJECT_SEX_01', 'APOE_num', 'simmilarity_bio' , 'simmilarity_conn', 'simmilarity_conn_fmri', 'BMI_z' , 'BPSYS_AVG_z' , 'BPDIA_AVG_z', 'DIABETES' ,  'ABratio' , 'PTAU181', 'NFL']]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Standardize the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Convert scaled data back to a dataframe for better handling
scaled_df = pd.DataFrame(scaled_data, columns=features.columns)



'''
# Generate the cluster map
sns.clustermap(scaled_df,
               method='ward',  # method for hierarchical clustering
               metric='euclidean',  # distance metric
               standard_scale=1,  # standardize the data across columns
               cmap='vlag',  # color map
               figsize=(10, 10))  # size of the plot
row_order = clustermap.dendrogram_row.reordered_ind
simmilarity_bio_reordered = filtered['simmilarity_bio'].values[row_order]


plt.title('Two-Way Hierarchical Clustering with Bars')
plt.show()
'''






# Create a clustermap
clustermap = sns.clustermap(scaled_df,
                            method='ward',  # method for hierarchical clustering
                            metric='euclidean',  # distance metric
                            standard_scale=1,  # standardize the data across columns
                            cmap='vlag',  # color map
                            figsize=(10, 10))

# Extract the order of the subjects (rows) from the clustermap
row_order = clustermap.dendrogram_row.reordered_ind

# Reorder the 'simmilarity_bio' column based on the clustering result
risk_scores = filtered['similarity_all'].values[row_order]



heatmap_ax = clustermap.ax_heatmap

# Create a new axis on the right side for the bar plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(heatmap_ax)
cax = divider.append_axes("right", size="7%", pad=0.05)

# Plot the bars on the new axis
cax.barh(np.arange(len(risk_scores)), risk_scores, color='red', edgecolor='none')
cax.set_yticks([])  # Remove y-axis ticks
cax.set_ylim(heatmap_ax.get_ylim())  # Match the ylim of the heatmap
cax.set_xlabel('personalized_risk')

plt.show()







'''
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering
linked = linkage(features, method='ward')

# Plot the dendrogram

# Create custom labels for each leaf node
labels = [f"Age: {age}, Sex: {'M' if sex == 0 else 'F'}, APOE: {apoe}" 
          for age, sex, apoe in zip(filtered['SUBJECT_AGE_SCREEN'], filtered['SUBJECT_SEX_01'], filtered['APOE'])]

# Plot the dendrogram with custom labels
plt.figure(figsize=(15, 10))
dendrogram(linked,
           orientation='top',
           labels=labels,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.xticks(rotation=90)
plt.savefig('/Users/ali/Desktop/Jul24/risk_new_from_June/plot/temp2/hierarchal.png', dpi=300)
plt.show()




linkage_matrix = linkage(features, method='ward')

# Create a dataframe for seaborn clustermap
clustermap_data = pd.DataFrame(filtered, columns=['SUBJECT_AGE_SCREEN_z', 'SUBJECT_SEX_01', 'APOE_num'])
clustermap_data['Cluster'] = clusters



palette = sns.color_palette("viridis", 5)
cluster_colors = [palette[label] for label in clusters]


# Create a clustermap with both hierarchical clustering and K-means clusters
sns.clustermap(clustermap_data.drop('Cluster', axis=1), row_cluster=True, col_cluster=False, 
               row_linkage=linkage_matrix, cmap='viridis', 
               row_colors=cluster_colors)

plt.title('Hierarchical Clustering with K-means Cluster Annotation')
plt.show()
'''

