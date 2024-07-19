#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:11:42 2024

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

# Function to z-score a single matrix
def z_score_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)
    z_scored_matrix = (matrix - mean) / std_dev
    return matrix
    #return z_scored_matrix
    
    
# Directory containing the CSV files
directory = "/Users/ali/Desktop/May24/Alex_Wstein_Risk_model_spline/connectome_act/"

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
            subj_list.append(int(os.path.basename(file_path)[2:6]))
            matrices.append(z_score_matrix(subj_df.to_numpy()))
df_conn_plain = pd.DataFrame(index=subj_list)
df_conn_plain['matrices']  = matrices
    


df_metadata = pd.read_csv("/Users/ali/Desktop/May24/Alex_Wstein_Risk_model_spline/AD_DECODE_data4.csv", encoding='utf-8')
df_metadata.set_index('MRI_Exam', inplace=True)


df = df_conn_plain.join(df_metadata, how='inner')


# Function to calculate Wasserstein distance between two matrices
def calculate_distance(matrix1, matrix2):
    return wasserstein_distance(matrix1.flatten(), matrix2.flatten())

def matrix_of_distances(matrices_a, matrices_b):
    distances = np.zeros((len(matrices_a), len(matrices_b)))
    for i, matrix_a in enumerate(matrices_a):
        for j, matrix_b in enumerate(matrices_b):
            distances[i, j] = calculate_distance(matrix_a, matrix_b)
    return distances
    

# Filter rows where 'risk_for_ad' is 2 or 3
risky = df[df['risk_for_ad'].isin([2, 3])]
print(risky)






df_distances = pd.DataFrame(matrix_of_distances(df['matrices'],risky['matrices']),index=df.index, columns=risky.index)



df_distances
df_distances.to_csv('df_distances.csv')


df_distances['average_distance'] = df_distances.mean(axis=1)
df = df.join(df_distances['average_distance'])
df['genotype_grouped'] = df['genotype'].map({'APOE23':'APOE23/33', 'APOE33':'APOE23/33', 'APOE34':'APOE34/44','APOE44':'APOE34/44'})

df

filtered = df[df['risk_for_ad'].isin([0,1])]

px.scatter(filtered, x='age', y=1/filtered.average_distance, size=filtered.risk_for_ad+1, color='genotype_grouped', template='none', trendline='ols', title='Age vs 1/distance')


# plot with the spline
filtered = filtered.sort_values('age')


spl = UnivariateSpline(filtered.age, 1/filtered.average_distance,k=3)
plt.plot(filtered.age, spl(filtered.age), 'g', lw=3)
colors = {"APOE23/33":"blue","APOE34/44":"red"}
plt.scatter(filtered.age, 1/filtered.average_distance, c=filtered['genotype_grouped'].map(colors))
plt.ylabel('1/Wasserstein Scaled')
plt.xlabel('Age')
plt.title('Linear Fit')
plt.legend()
plt.show()

simmilarity=(1/filtered.average_distance)/max(1/filtered.average_distance)
spl = UnivariateSpline(filtered.age, simmilarity,k=3)
plt.plot(filtered.age, spl(filtered.age), 'g', lw=3)
plt.scatter(filtered.age, simmilarity, c=filtered['genotype_grouped'].map(colors))
plt.ylabel('Simmilarity')
plt.xlabel('Age')
plt.title('Linear Fit')
plt.legend()
plt.show()



df['age_corrected_risk'] = 1/df.average_distance - spl(df.age)
filtered = df[df['risk_for_ad'].isin([0,1])]

px.scatter(df, x='age', y='age_corrected_risk', color='genotype_grouped', template='none', title='Age corrected risk')

px.scatter(df, x='age', y='age_corrected_risk', color='risk_for_ad', template='none', title='Age corrected risk')

px.box(df, x='risk_for_ad', y='age_corrected_risk', template='none', title='Age corrected risk')

px.box(filtered, x='sex', y='age_corrected_risk', template='none', title='Age corrected risk')

px.box(filtered, x='genotype_grouped', y='age_corrected_risk', template='none', title='Age corrected risk by genotype')


youngest = df.sort_values('age').groupby('Family').head(1)
oldest = df.sort_values('age').groupby('Family').tail(1)
families = youngest.merge(oldest, on='Family', suffixes=('_youngest', '_oldest'))

px.scatter(families, x='age_corrected_risk_youngest', y='age_corrected_risk_oldest', template='none', trendline='ols', title='Youngest vs Oldest family members')

px.scatter(families, x='average_distance_youngest', y='average_distance_oldest', template='none', trendline='ols', title='Youngest vs Oldest family members')

px.scatter(filtered, x='Systolic', y='average_distance', color='genotype_grouped', template='none', trendline='ols', title='Age vs Distance')


################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin


class CustomSplineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_knots=3, spline_degree=3):
        self.n_knots = n_knots
        self.spline_degree = spline_degree
    
    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        
        # Sort X and y
        sorted_indices = np.argsort(self.X_train_)
        self.sorted_X_ = self.X_train_[sorted_indices]
        self.sorted_y_ = self.y_train_[sorted_indices]
        
        # Remove duplicates and average corresponding y values
        unique_sorted_X, unique_indices = np.unique(self.sorted_X_, return_index=True)
        unique_sorted_y = [np.mean(self.sorted_y_[self.sorted_X_ == ux]) for ux in unique_sorted_X]
        unique_sorted_y = np.array(unique_sorted_y)
        
        if len(unique_sorted_X) < self.n_knots + self.spline_degree + 1:
            raise ValueError("Not enough unique points to form a spline with the given number of knots and degree.")
        
        # Choose knot points
        knot_indices = np.linspace(0, len(unique_sorted_X) - 1, self.n_knots, dtype=int)
        self.knot_points_ = unique_sorted_X[knot_indices]
        
        # Internal knots (excluding first and last)
        t = np.r_[[unique_sorted_X[0]] * self.spline_degree, self.knot_points_, [unique_sorted_X[-1]] * self.spline_degree]
        
        # Fit spline
        self.spline_ = make_lsq_spline(unique_sorted_X, unique_sorted_y, t, k=self.spline_degree)
        
        return self
    
    def predict(self, X):
        return self.spline_(X)
    
    
class CustomSplineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_knots=3, spline_degree=3):
        self.n_knots = n_knots
        self.spline_degree = spline_degree
    
    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        
        # Sort X and y
        sorted_indices = np.argsort(X)
        self.sorted_X_ = X[sorted_indices]
        self.sorted_y_ = y[sorted_indices]
        
        # Choose knot points
        knot_indices = np.linspace(0, len(X) - 1, self.n_knots, dtype=int)
        self.knot_points_ = self.sorted_X_[knot_indices]
        
        # Fit spline
        self.spline_ = UnivariateSpline(self.sorted_X_, self.sorted_y_, k=self.spline_degree)
        
        return self
    
    def predict(self, X):
        return self.spline_(X)
    
    

# Generate synthetic data resembling logistic regression
np.random.seed(0)
X_train = np.linspace(0, 100, 100)  # Generate 100 evenly spaced points from 0 to 100 for X
logistic_function = lambda x: 100 * np.exp((x - 50) / 10) / (1 + np.exp((x - 50) / 10))  # Adjusted logistic function to have inflection point at 50 and asymptotically reach 100
y_train = logistic_function(X_train) + np.random.normal(0, 0.1, size=100)  # Add some noise

# Fit the custom spline model
custom_spline = CustomSplineRegressor(n_knots=3, spline_degree=3)
custom_spline = UnivariateSpline(X_train, y_train,k=3)

# Define two points that we want the spline to fit
P1 = np.array([20, 50])  # Example point 1
P2 = np.array([80, 75])  # Example point 2

# Evaluate the spline at specific x values
S_t1 = np.array([P1[0], custom_spline([P1[0]])[0]])
S_t2 = np.array([P2[0], custom_spline([P2[0]])[0]])

# Calculate the translation vector T
T = 0.5 * ((P1 - S_t1) + (P2 - S_t2))

# Translate the spline
def translated_spline(x):
    #return custom_spline.predict(x) + T[1]
    return custom_spline(x) + T[1]

# Plot the original and translated splines
x_range = np.linspace(min(X_train), max(X_train), 100)
original_spline_values = custom_spline(x_range)
translated_spline_values = translated_spline(x_range)

plt.scatter(X_train, y_train, label='Original Data')
plt.plot(x_range, original_spline_values, color='blue', label='Original Spline')
plt.plot(x_range, translated_spline_values, color='red', label='Translated Spline')
plt.scatter([P1[0], P2[0]], [P1[1], P2[1]], color='green', zorder=5, label='Target points')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Custom Spline Translation to Fit Two Points')
plt.show()

#####


from matplotlib.pyplot import cm

df['average_distance'] = 1 / df['average_distance']
#df['average_distance'] = 100*(df['average_distance'] - np.min(df['average_distance'])) / (np.max(df['average_distance'])-np.min(df['average_distance']))
df['average_distance'] = (df['average_distance']) / (np.max(df['average_distance']))





X_train = df['age'].to_numpy()
y_train = df['average_distance'].to_numpy()

X_train_argsort = X_train.argsort()

X_train = X_train[X_train_argsort]
y_train = y_train[X_train_argsort]
# Fit the custom spline model
custom_spline = CustomSplineRegressor(n_knots=3, spline_degree=3)
custom_spline =  UnivariateSpline(X_train, y_train, k=3)
x_range = np.linspace(min(X_train), max(X_train), 100)
custom_spline(x_range)
#y_train = 100*(y_train - np.min(custom_spline.predict(x_range))) / (np.max(custom_spline.predict(x_range))-np.min(custom_spline.predict(x_range)))



fami_uniq= df['Fam_Num'].dropna().unique()
color = cm.rainbow(np.linspace(0, 1, len(fami_uniq)))
for enum, family in enumerate(fami_uniq):
    #family = fami_uniq[3]
    dftemp = df[df['Fam_Num']==family]

    
# Define two points that we want the spline to fit
    P1 = np.array( [np.asarray( list(dftemp['age']))[0],np.asarray( list(dftemp['average_distance']))[0] ] )  # Example point 1
    P2 = np.array( [np.asarray( list(dftemp['age']))[1],np.asarray( list(dftemp['average_distance']))[1] ] )  # Example point 1
    S_t1 = np.array([P1[0], spl([P1[0]])[0]])
    S_t2 = np.array([P2[0], spl([P2[0]])[0]])
    T = 0.5 * ((P1 - S_t1) + (P2 - S_t2))
    # Translate the spline
    def translated_spline(x):
        return(spl(x)  + T[1])
# Plot the original and translated splines
    x_range = np.linspace(min(X_train), max(X_train), 100)
    #original_spline_values = custom_spline.predict(x_range) 
    
    translated_spline_values = translated_spline(x_range)

#plt.scatter(X_train, y_train, label='Original Data')
#plt.plot(x_range, original_spline_values, color='blue', label='Original Spline')
    plt.plot(x_range, translated_spline_values, color=color[enum,:], label=' '+ str(family) )
    plt.scatter([P1[0], P2[0]], [P1[1], P2[1]], color=color[enum,:], zorder=5)
    plt.legend(loc=1, mode='expand', numpoints=1, ncol=4, fancybox = True,
           fontsize='small')
    #plt.yscale()
    plt.xlabel('Age (year)')
    plt.ylabel('1/dist')
#plt.legend()
#plt.title('Custom Spline Translation to Fit Two Points')
plt.show()
