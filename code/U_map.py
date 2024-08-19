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



import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns




# Convert the columns with lists/arrays to a 2D numpy array
# Assuming 'vectorized_matrices_plain' and 'matrices_fmri' are lists of the same length
features_plain = np.array([np.ravel(x) for x in df['matrices_plain']])
features_fmri = np.array([np.ravel(x) for x in df['matrices_fmri']])


# Concatenate the features along the second axis (columns)
features = np.concatenate([features_plain, features_fmri], axis=1)



reducer = umap.UMAP()
embedding = reducer.fit_transform(features)



# Visualize the UMAP embedding
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['SUBJECT_AGE_SCREEN'], cmap='viridis', s=5)
plt.colorbar(scatter, label='Age')
plt.title('UMAP projection of the data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



# Plot with sex coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['SUBJECT_SEX'].map({1: 0, 2: 1}), cmap='coolwarm', s=5)
plt.colorbar(scatter, label='Sex (0: M, 1: F)')
plt.title('UMAP projection of the data (colored by sex)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



# Plot with APOE status coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['APOE'].map({'2/4': 1, '3/3': 0, '3/4': 1, '4/4' : 1}), cmap='plasma', s=5)
plt.colorbar(scatter, label='APOE status')
plt.title('UMAP projection of the data (colored by APOE status)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



df['APOE_num'] = df['APOE'].map({'2/4': 1, '2/3':0, '3/3': 0, '3/4': 1, '4/4' : 1})
demographic_features = df[['SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'APOE_num']]

# Apply UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(demographic_features)




# Visualize the UMAP embedding
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['SUBJECT_AGE_SCREEN'], cmap='viridis', s=5)
plt.colorbar(scatter, label='Age')
plt.title('UMAP projection of the data (based on demographic data)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()




# Plot with sex coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['SUBJECT_SEX'], cmap='coolwarm', s=5)
plt.colorbar(scatter, label='Sex (0: M, 1: F)')
plt.title('UMAP projection of the data (colored by sex)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()




# Plot with APOE status coloring
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['APOE_num'], cmap='plasma', s=5)
plt.colorbar(scatter, label='APOE status')
plt.title('UMAP projection of the data (colored by APOE status)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



'''
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

simmilarity_bio=  (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
simmilarity_conn = (1/filtered.average_distance)/max(1/filtered.average_distance)
simmilarity_conn_fmri = (1/filtered.average_distance_fmri)/max(1/filtered.average_distance_fmri)

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


filteredreg = filtered
filteredreg.dropna(subset=['LTPR'], inplace=True)
y = filteredreg['LTPR']
y= (y-np.mean(y) ) / np.std(y)

X = filteredreg.simmilarity_conn.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_dti = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_dti)
print("Average RMSE simmilarity_conn:", average_rmse)
sd_rmse = np.std(rmse_scores_dti)
print("SD RMSE simmilarity_conn:", sd_rmse)


X = filteredreg.similarity_dt_fm.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_fm_dt = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_fm_dt)
print("Average RMSE similarity_dt_fm:", average_rmse)
sd_rmse = np.std(rmse_scores_fm_dt)
print("SD RMSE:", sd_rmse)

X = filteredreg.similarity_dt_bio.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_dti_bio = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_dti_bio)
print("Average RMSE similarity_dt_bio:", average_rmse)
sd_rmse = np.std(rmse_scores_dti_bio)
print("SD RMSE:", sd_rmse)

X = filteredreg.similarity_bio_fm.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_fm_bio = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_fm_bio)
print("Average RMSE similarity_bio_fm:", average_rmse)
sd_rmse = np.std(rmse_scores_fm_bio)
print("SD RMSE:", sd_rmse)


X = filteredreg.simmilarity_bio.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_bio = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_bio)
print("Average RMSE simmilarity_bio:", average_rmse)
sd_rmse = np.std(rmse_scores_bio)
print("SD RMSE:", sd_rmse)

X = filteredreg.similarity_all.values.reshape(-1, 1) 
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
rmse_scores_all = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
average_rmse = np.mean(rmse_scores_all)
print("Average RMSE similarity_all:", average_rmse)
sd_rmse = np.std(rmse_scores_all)
print("SD RMSE:", sd_rmse)





# Function to remove outliers
def remove_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if x >= lower_bound and x <= upper_bound]

cleaned_rmse_scores_dti = remove_outliers(rmse_scores_dti)
cleaned_rmse_scores_fm_dt = remove_outliers(rmse_scores_fm_dt)
cleaned_rmse_scores_dti_bio = remove_outliers(rmse_scores_dti_bio)
cleaned_rmse_scores_fm_bio = remove_outliers(rmse_scores_fm_bio)
cleaned_rmse_scores_bio = remove_outliers(rmse_scores_bio)
cleaned_rmse_scores_all = remove_outliers(rmse_scores_all)


# Plotting
plt.figure(figsize=(10, 10))
plt.boxplot([cleaned_rmse_scores_dti, cleaned_rmse_scores_fm_dt, cleaned_rmse_scores_dti_bio, cleaned_rmse_scores_fm_bio, cleaned_rmse_scores_bio, cleaned_rmse_scores_all], 
            vert=False, 
            patch_artist=True, 
            labels=['DTI', 'fMRI+DTI', 'DTI+Biomarker', 'fMRI+Biomarker', 'Biomarker' , 'All'],  widths=0.1, positions=[0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.title('Boxplot of LTPR RMSE')
plt.xlabel('RMSE')
plt.ylabel('Variables')
plt.grid(True)
plt.savefig('/Users/ali/Desktop/Jul24/risk_new_from_June/plot/RMSE_LTPR.png', dpi=300)
plt.show()
'''