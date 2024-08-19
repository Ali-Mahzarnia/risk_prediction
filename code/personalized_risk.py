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
    return z_scored_matrix
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
#directory = '/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/connectome/DTI/plain/'
directory ='/Users/ali/Desktop/Jul24/ADRC_connectome/DTI_plain/'

subj_list = []
matrices = []
matrix_size = (84,84)

# Iterate through files in the directory
for filename in os.listdir(directory):
    #if re.search("\d_conn_plain.csv$",filename):
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
    
#dir_fmri = '/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/connectome/fMRI'
dir_fmri ='/Users/ali/Desktop/Jul24/ADRC_connectome/fmri/'
subj_list = []
matrices = []
matrix_size = (84,84)

# Iterate through files in the directory
for filename in os.listdir(dir_fmri):
    if re.search("func_connectome_covar_",filename):
        file_path = os.path.join(dir_fmri, filename)
        # Read CSV file into a dataframe
        subj_df = pd.read_csv(file_path, header=None)  # Assuming there is no header
        if subj_df.shape == matrix_size:
            subj_list.append(int(os.path.basename(file_path)[26:30]))
            z_scored_matrix = z_score_matrix(subj_df.to_numpy())
            thresholded_matrix = threshold_matrix(z_scored_matrix, 10)
            matrices.append(thresholded_matrix)
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
df_metadata ['ABratio_orig'] = df_metadata ['ABratio']
df_metadata ['ABratio']  =  z_score_matrix( df_metadata ['ABratio'] )
df_metadata ['PTAU181_orig'] = df_metadata ['PTAU181'] 
df_metadata ['PTAU181']  = z_score_matrix( df_metadata ['PTAU181'] )
df_metadata ['NFL_orig'] = df_metadata ['NFL']
df_metadata ['NFL']  =  z_score_matrix( df_metadata ['NFL'] )




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

#metric_fun( X[1],X[2])
###############


X = np.array( [matrix.flatten() for matrix in df.matrices_fmri] )
y = np.array( df.DEMENTED)



nca = NCA(random_state=2)
nca.fit(X, y)

metric_fun3 = nca.get_metric()

#metric_fun( X[1],X[2])






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
filtered['simmilarity_dti'] = simmilarity_conn
filtered['simmilarity_fmri'] = simmilarity_conn_fmri
filtered['similarity_dt_fm'] = similarity_dt_fm
filtered['similarity_dt_bio'] = similarity_dt_bio
filtered['similarity_bio_fm'] = similarity_bio_fm
filtered['similarity_all'] = similarity_all





distance_matrix = np.zeros((len(filtered), len(filtered)))


num_rows = len(filtered)
for i in range(num_rows):
    row_i = filtered.iloc[i]
    for j in range(num_rows):
        row_j = filtered.iloc[j]
        dist_plain = metric_fun(row_i['matrices_plain'].flatten(), row_j['matrices_plain'].flatten())
        dist_fmri = metric_fun3(row_i['matrices_fmri'].flatten(), row_j['matrices_fmri'].flatten())
        biom1 = row_i[['ABratio', 'PTAU181', 'NFL']].values
        biom2 = row_j[['ABratio', 'PTAU181', 'NFL']].values
        dist_biom = metric_fun2(biom1, biom2)
        distance_matrix[i, j] = (dist_plain + dist_fmri + dist_biom) / 3
        #print(f"i : {i} , j: {j}")

print(distance_matrix)



#########

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


apoe_mapping = {'3/3': 0, '3/4': 1, '4/4': 1,'2/3': 0, '2/4': 1 }
filtered['APOE_mapped'] = filtered['APOE'].map(apoe_mapping)


results = []
#results2=[]

for i in range(num_rows):
    distances = distance_matrix[i]
    sorted_indices = np.argsort(distances)
    
    # Filter healthy and demented indices
    healthy_indices = [idx for idx in sorted_indices if filtered.iloc[idx]['DEMENTED'] == 0 and not np.isnan(distances[idx])]
    demented_indices = [idx for idx in sorted_indices if filtered.iloc[idx]['DEMENTED'] == 1 and not np.isnan(distances[idx])]
    
    # Select the 10 closest healthy and 10 closest demented subjects
    closest_healthy_indices = healthy_indices[:10]
    closest_demented_indices = demented_indices[:10]
    
    # Combine indices
    combined_indices = closest_healthy_indices + closest_demented_indices
    
    # Ensure the subject itself is not included in the training data
    combined_indices = [idx for idx in combined_indices if idx != i]

    # Prepare data for logistic regression
    X_train = filtered.iloc[combined_indices][['APOE_mapped', 'SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'simmilarity_dti','simmilarity_fmri' , 'DIABETES', 'ABratio', 'PTAU181', 'NFL']]
    y_train = filtered.iloc[combined_indices]['DEMENTED']
    
    # Fit logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict the likelihood of the subject being unhealthy
    X_test = filtered.iloc[[i]][['APOE_mapped', 'SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'simmilarity_dti','simmilarity_fmri','DIABETES', 'ABratio', 'PTAU181', 'NFL']]
    likelihood = model.predict_proba(X_test)[:, 1]  # Probability of being demented
    coefficients = model.coef_[0]
    # Store the result
    results.append((i, likelihood[0], coefficients))
    '''
# Append the result for plotting

# Extract ages and likelihoods for the training data
    ages_train = X_train['SUBJECT_AGE_SCREEN']
    likelihoods_train = model.predict_proba(X_train)[:, 1]

# Extract age and likelihood for the test subject
    age_test = X_test['SUBJECT_AGE_SCREEN'].values[0]

# Generate a range of ages for plotting the logistic curve
    age_range = np.linspace(ages_train.min(), ages_train.max(), 300)
    X_range = np.tile(X_train.mean(axis=0), (300, 1))
    X_range[:, X_train.columns.get_loc('SUBJECT_AGE_SCREEN')] = age_range

# Predict likelihoods for the age range
    likelihoods_range = model.predict_proba(X_range)[:, 1]

# Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(ages_train, likelihoods_train, color='blue', label='Training Data')
    plt.scatter(age_test, likelihood, color='red', label='Test Data', edgecolor='k')
    plt.plot(age_range, likelihoods_range, color='green', label='Logistic Regression Curve')
    plt.xlabel('Subject Age')
    plt.ylabel('Likelihood of being demented')
    plt.title('Logistic Regression: Likelihood of Being Demented vs Age')
    plt.legend()
    plt.show()
    '''
    
    #results2.append((likelihood[0], X_test.values, filtered.iloc[i]['DEMENTED'] ))
    
    print(f"i={i} , DEM={filtered.iloc[i]['DEMENTED']} , Liklihood = {likelihood}, Age = {X_test['SUBJECT_AGE_SCREEN'].values}")



# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results, columns=['Subject_Index', 'Likelihood_of_Being_Unhealthy', 'Coefficients'])
#results_df2 = pd.DataFrame(results2, columns=['Risk', 'data' ,'DEMENTED' ])



print(results_df)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Assuming `results_df` and `filtered` DataFrames are already created from the previous code

# Extract the coefficients for each subject
coefficients_df = pd.DataFrame(results_df['Coefficients'].tolist(), columns=['APOE_mapped', 'SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'simmilarity_dti','simmilarity_fmri', 'DIABETES' , 'ABratio', 'PTAU181', 'NFL'])

# Reset index of filtered to ensure alignment
filtered.reset_index(drop=True, inplace=True)
coefficients_df.reset_index(drop=True, inplace=True)


# Concatenate the input variables and the coefficients
combined_df = pd.concat([filtered[[]], coefficients_df], axis=1)



# Standardize the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_df)

# Convert scaled data back to a DataFrame for better handling
scaled_df = pd.DataFrame(scaled_data, columns=combined_df.columns)

# Create a clustermap
clustermap = sns.clustermap(scaled_df,
                            method='ward',  # method for hierarchical clustering
                            metric='euclidean',  # distance metric
                            standard_scale=1,  # standardize the data across columns
                            cmap='vlag',  # color map
                            figsize=(10, 10))

# Extract the order of the subjects (rows) from the clustermap
row_order = clustermap.dendrogram_row.reordered_ind

# Reorder the 'similarity_all' column based on the clustering result
risk_scores = results_df['Likelihood_of_Being_Unhealthy'].values[row_order]


heatmap_ax = clustermap.ax_heatmap

# Create a new axis on the right side for the bar plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(heatmap_ax)
cax = divider.append_axes("left", size="5%", pad=0.0)



# Plot the bars on the new axis
cax.barh(np.arange(len(risk_scores)), risk_scores, color='red', edgecolor='none')
cax.set_yticks([])  # Remove y-axis ticks
cax.set_ylim(heatmap_ax.get_ylim())  # Match the ylim of the heatmap
cax.set_xlabel('personalized_risk')
plt.savefig('/Users/ali/Desktop/Jul24/risk_new_from_June/plot/temp2/hierarchal.png', dpi=300)
plt.show()




import pandas as pd

# Assuming `filtered` and `results_df` DataFrames are already created from the previous code

# Extract the relevant columns
data_columns = filtered[['id' , 'APOE_mapped','APOE' , 'SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'simmilarity_dti', 'simmilarity_fmri', 'DIABETES', 'ABratio', 'PTAU181', 'NFL','ABratio_orig', 'PTAU181_orig', 'NFL_orig', 'DEMENTED']]

# Reset index to ensure alignment
data_columns.reset_index(drop=True, inplace=True)

# Extract the order of the subjects (rows) from the clustermap
row_order = clustermap.dendrogram_row.reordered_ind

# Reorder the data_columns DataFrame based on the clustering result
reordered_data_columns = data_columns.iloc[row_order]

# Add the risk_scores to the reordered DataFrame
reordered_data_columns['personalized_risk'] = risk_scores

# Create a DataFrame for risk_scores and coefficients
coefficients_df = pd.DataFrame(results_df['Coefficients'].tolist(), columns=['APOE_mapped', 'SUBJECT_AGE_SCREEN', 'SUBJECT_SEX', 'simmilarity_dti', 'simmilarity_fmri', 'DIABETES', 'ABratio', 'PTAU181', 'NFL'])
risk_and_coefficients_df = pd.concat([pd.Series(risk_scores, name='personalized_risk'), coefficients_df], axis=1)

# Write both DataFrames to an Excel file with multiple sheets
output_excel_path = '/Users/ali/Desktop/Jul24/risk_new_from_June/plot/temp2/personalized_risk_data.xlsx'
with pd.ExcelWriter(output_excel_path) as writer:
    reordered_data_columns.to_excel(writer, sheet_name='Data_with_Risk_Scores', index=False)
    risk_and_coefficients_df.to_excel(writer, sheet_name='Risk_and_Coefficients', index=False)

print(f"Data with personalized risk scores and coefficients has been written to {output_excel_path}")
