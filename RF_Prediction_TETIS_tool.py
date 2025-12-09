# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:58:23 2025

@author: ncortor
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
from tqdm import tqdm

__author__ = 'Ing. MSc.  PhD(c) Nicolás Cortés-Torres'
__copyright__ = "Copyright 2024, NCT"
__credits__ = ["Nicolás Cortés-Torres"]
__license__ = "GIMHA"
__version__ = "0.1"
__maintainer__ = "Nicolás Cortés-Torres"
__email__ = 'ncortor@doctor.upv.es, ingcortest@gmail.com'
__status__ = "developing"

#%%######## FUNCTIONS ################################################################################

################################################################################
# Function to set the current time
def def_hora():
    return time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
   
################################################################################
def rmse(obs, sim):
    """ Root mean squared error
    This function calculates the square root of the mean square error between the observed data and the simulated data.
    :obs: The observed data is arranged in columns.
    :sim: The simulated data is arranged in columns.
    :return: Vector with the square root of the mean square error between the observed data and the simulated data
    """
    n = len(obs)
    return np.sqrt(np.nansum(np.power(obs - sim, 2), axis=0) / n)

################################################################################
def rsqr(obs, sim):
    """ Coefficient of determination
    This function calculates the coefficient of determination between the observed data and the simulated data.
    :obs: The observed data is arranged in columns.
    :sim: The simulated data is arranged in columns.
    :return: Vector with the coefficient of determination between the observed data and the simulated data
    """
    a = obs - np.nanmean(obs, axis=0)
    b = sim - np.nanmean(sim, axis=0)
    return np.power((np.nansum(a * b, axis=0) / np.sqrt(np.nansum(np.power(a, 2), axis=0) * np.nansum(np.power(b, 2), axis=0))), 2)
    
#%%#############################################################################################################
############################### Prediction Tool for excution times on TETIS ####################################
################################################################################################################

tic = time.time()           # Start timer for script execution tracking
script_dir = Path.cwd()     # Get the current working directory (script's location)
print(f"✅ Reading User predictor variables --- {def_hora()} ---")

# --- Load User Input Data ---
PV_path = script_dir / "User_predictor_variables.csv"   # Construct the path to the input CSV file
PV = pd.read_csv(PV_path, encoding="utf-8")             # Load user input variables into a DataFrame

# --- Read and cast user inputs from the DataFrame ---
speed= float(PV.iloc[0, 1])                                     # Max turbo frequency
area_km2= float(PV.iloc[1, 1])                                  # Basin area in square kilometers
cell_size_m = float(PV.iloc[2, 1])                              # Cell size in meters
ini = pd.to_datetime(PV.iloc[3, 1], format='%d/%m/%Y %H:%M')    # Initial date
fin = pd.to_datetime(PV.iloc[4, 1], format='%d/%m/%Y %H:%M')    # Final date
dt = float(PV.iloc[5, 1])                                         # Delta t in minutes
n_in = float(PV.iloc[6, 1])                                       # Number of input gauges
n_out = float(PV.iloc[7, 1])                                      # Number of output gauges
ram = float(PV.iloc[8, 1])                                        # RAM memory capacity
n_core = float(PV.iloc[9, 1])                                     # Number of cores
n_thr = float(PV.iloc[10, 1])                                     # Number of threads

# --- Calculate Derived Predictor Variables ---
cell_area = ((cell_size_m*cell_size_m)/1000000)     # Calculate area of a single cell in square kilometers
cells = area_km2 / cell_area                        # Estimate the total number of cells in the basin (spatial complexity)
timesteps = (fin - ini).total_seconds() / (60*dt)   # Calculate total number of time steps (temporal complexity)

# --- Define Predictor Variable Vectors for Two Models (Topolco & Sim) ---

# Vector with Topolco predictor variables
Top_pv = pd.DataFrame([speed, cells, ram, n_core, n_thr], 
                     index = ['Max turbo frequency', 'Basin cells', 'RAM memory', 'Cores', 'Threads']).T

# Vector with Hydrological Simulation (HS) predictor variables
Sim_pv = pd.DataFrame([speed, cells, timesteps, n_in, n_out],
                      index = ['Max turbo frequency', 'Basin cells', 'Time steps', 'Input gauges', 'Output gauges']).T

# --- Data Validation: Check for Missing Values ---
# Check which columns (predictor variables) have no NaN values
Top_no_nan = Top_pv.columns[Top_pv.notna().all(axis=0)].tolist()
Sim_no_nan = Sim_pv.columns[Sim_pv.notna().all(axis=0)].tolist()

# Filter predictor variables by non-NaN columns (use only available variables)
Top_pv_user = Top_pv[Top_no_nan]
Sim_pv_user = Sim_pv[Sim_no_nan]

# --- Read Hyperparameters for Random Forest (RF) Models ---
# The hyperparameters are needed to load/configure the pre-trained RF models

Top_HiperP_path = script_dir / "Topolco_Hyperparameter.pickle"      # Path to Topolco hyperparameters file
Top_hiperp = pd.read_pickle(Top_HiperP_path)                        # Load hyperparameters for the Topolco model

Sim_HiperP_path = script_dir / "Tetis_Hyperparameter.pickle"        # Path to Hydrological Sim hyperparameters file
Sim_hiperp = pd.read_pickle(Sim_HiperP_path)                        # Load hyperparameters for the Hydrological Sim model

# --- Search for Matching Hyperparameter Combination ---
# Create searchable strings of available predictor variables for each model
Top_to_search = ", ".join(Top_no_nan)
Sim_to_search = ", ".join(Sim_no_nan)

# Filter the loaded Hyperparameter DataFrame to find a row where the combination of indices EXACTLY matches the available variables
Top_hiper_selec = Top_hiperp[Top_hiperp['Combinacion_Indices'] == Top_to_search]
Sim_hiper_selec = Sim_hiperp[Sim_hiperp['Combinacion_Indices'] == Sim_to_search]

# --- Log Hyperparameter Search Results ---
if not Top_hiper_selec.empty:
    print("\n✅ Topolco hiperparameters found!")     # Success: Matching hyperparameters found
else:
    print(f"\n❌ No Topolco hyperparameters were found for the combination of given predictor variables") # Failure: No match found

if not Sim_hiper_selec.empty:
    print("✅ Hydrological simulation hiperparameters found!\n")    # Success: Matching hyperparameters found
else:
    print(f"❌ No Hydrological simulation hyperparameters were found for the combination of given predictor variables\n") # Failure: No match found

# --- Allocate Hyperparameters for RF Model Configuration ---
# These parameters (e.g., n_estimators, max_depth) are extracted from the matched row (index 0) of the hyperparameters dataframes

# Topolco RF Model Hyperparameters
top_estimator = int(Top_hiper_selec.iloc[0, 8])     # Number of trees in the forest (n_estimators)
top_depth = int(Top_hiper_selec.iloc[0, 9])         # Maximum depth of the tree (max_depth)
top_split = int(Top_hiper_selec.iloc[0, 10])        # Minimum number of samples required to split an internal node (min_samples_split)
top_leaf =  int(Top_hiper_selec.iloc[0, 11])        # Minimum number of samples required to be at a leaf node (min_samples_leaf)

# Hydrological Simulation RF Model Hyperparameters
sim_estimator = int(Sim_hiper_selec.iloc[0, 8])     # Number of trees in the forest (n_estimators)
sim_depth = int(Sim_hiper_selec.iloc[0, 9])         # Maximum depth of the tree (max_depth)
sim_split = int(Sim_hiper_selec.iloc[0, 10])        # Minimum number of samples required to split an internal node (min_samples_split)
sim_leaf =  int(Sim_hiper_selec.iloc[0, 11])        # Minimum number of samples required to be at a leaf node (min_samples_leaf)

# --- Read Training Data for RF Model Prediction ---
# Load the historical data used to train the RF models

data_sim_path = script_dir / "Database_times_tetis.pickle"  # Path to Hydrological Sim training data
data_sim = pd.read_pickle(data_sim_path)                    # Load training data

data_top_path = script_dir / "Database_times_topolco.pickle"    # Path to Topolco training data
data_top = pd.read_pickle(data_top_path)                        # Load training data

# --- Prepare Data for RF Model Input ---
# Filter and extract the predictor variables (X) and target variable (Y) arrays

# Prepare predictor features (X) using only non-NaN columns
df_sim_X = data_sim[Sim_no_nan].values
df_top_X = data_top[Top_no_nan].values

# Prepare target variable (Y) arrays (target is assumed to be at a fixed column index in the original training dataframes)
df_sim_Y = data_sim.iloc[:, 7].values
df_top_Y = data_top.iloc[:, 6].values


#%%########### --- Prediction Application (Monte Carlo Simulation) ---

# Define the number of iterations for the Monte Carlo simulation
n_iter=100
Top_times = [] # List to store prediction results for the Topolco model
Sim_times = [] # List to store prediction results for the Hydrological Simulation (Sim) model
    
# Loop for Monte Carlo simulation (100 iterations) with progress bar (tqdm)
for i in tqdm(range(0,n_iter),
              desc="Processing Random Forest models",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
    
    # --- TOPOLCO RF Model Training and Evaluation ---
    # Split training data into training and test sets for evaluation
    Top_X_train, Top_X_test, Top_Y_train, Top_Y_test = train_test_split(
    df_top_X, 
    df_top_Y.ravel(), 
    test_size= random.uniform(0.20,0.30),           # Randomly selects a test size between 20% and 30% for uncertainty
    random_state= random.randint(0, 4294967295)     # Random seed for reproducible, yet varying, splits
    )
    
    # Define the Random Forest Regressor algorithm using pre-determined hyperparameters
    Top_rf = RandomForestRegressor(n_estimators=top_estimator, 
                               max_depth=top_depth, 
                               min_samples_split=top_split, 
                               min_samples_leaf=top_leaf)# Definition of the algorithm

    Top_rf.fit(Top_X_train, Top_Y_train) # Train the RF algorithm on the Topolco training data
    Top_Y_sim = Top_rf.predict(Top_X_test) # Predict performance on the Topolco test set

    top_index_rmse = rmse(Top_Y_test, Top_Y_sim) # Calculate Root Mean Square Error (RMSE) for evaluation
    top_index_rsqr = rsqr(Top_Y_test, Top_Y_sim) # Calculate R-squared (R²) for evaluation

    # Predict the running time for the current user's inputs (Top_pv_user)
    Top_time_user = Top_rf.predict(Top_pv_user.values)
    
    # Store the predicted time and the model metrics for this iteration
    Top_times.append({
        'Time': Top_time_user[0],
        'rmse': top_index_rmse,
        'rsqr': top_index_rsqr
        })
        
    # --- Hydrological Simulation (Sim) RF Model Training and Evaluation ---
    Sim_X_train, Sim_X_test, Sim_Y_train, Sim_Y_test = train_test_split(
    df_sim_X, 
    df_sim_Y.ravel(), 
    test_size= random.uniform(0.20,0.30),           # Randomly selects a test size between 20% and 30%
    random_state= random.randint(0, 4294967295)     # Random seed for reproducible, yet varying, splits
    )
    
    # Define the Random Forest Regressor algorithm using pre-determined hyperparameters
    Sim_rf = RandomForestRegressor(n_estimators=sim_estimator, 
                               max_depth=sim_depth, 
                               min_samples_split=sim_split, 
                               min_samples_leaf=sim_leaf)# Definition of the algorithm

    Sim_rf.fit(Sim_X_train, Sim_Y_train)    # Train the RF algorithm on the Sim training data
    Sim_Y_sim = Sim_rf.predict(Sim_X_test)  # Predict performance on the Sim test set

    sim_index_rmse = rmse(Sim_Y_test, Sim_Y_sim) # Calculate Root Mean Square Error (RMSE) for evaluation
    sim_index_rsqr = rsqr(Sim_Y_test, Sim_Y_sim) # Calculate R-squared (R²) for evaluation

    # Predict the running time for the current user's inputs (Sim_pv_user)
    Sim_time_user = Sim_rf.predict(Sim_pv_user.values) # Calculate simulate Y 
    
    # Store the predicted time and the model metrics for this iteration
    Sim_times.append({
        'Time': Sim_time_user[0],
        'rmse': sim_index_rmse,
        'rsqr': sim_index_rsqr
        })
    pass    # End of the loop block

# Convert list of results to Pandas DataFrames for easy aggregation
df_Top_times= pd.DataFrame(Top_times)
df_Sim_times= pd.DataFrame(Sim_times)

# --- Final Aggregation and Calculation of Uncertainty Metrics (Topolco) ---
top_y_sim_mean = df_Top_times["Time"].mean()    # Mean predicted running time (Topolco)
top_y_sim_max = df_Top_times["Time"].max()      # Maximum predicted running time (Topolco)
top_y_sim_min = df_Top_times["Time"].min()      # Minimum predicted running time (Topolco)
top_mean_rmse = df_Top_times["rmse"].mean()     # Mean RMSE across all iterations
top_mean_rsqr = df_Top_times["rsqr"].mean()     # Mean R-squared across all iterations

# --- Final Aggregation and Calculation of Uncertainty Metrics (Hydrological Sim) ---
sim_y_sim_mean = df_Sim_times["Time"].mean()    # Mean predicted running time (Sim)
sim_y_sim_max = df_Sim_times["Time"].max()      # Maximum predicted running time (Sim)
sim_y_sim_min = df_Sim_times["Time"].min()      # Minimum predicted running time (Sim)
sim_mean_rmse = df_Sim_times["rmse"].mean()     # Mean RMSE across all iterations
sim_mean_rsqr = df_Sim_times["rsqr"].mean()     # Mean R-squared across all iterations

prediction = f"""
=====================================================
||     RUNTIME PREDICTIONS FOR TETIS V9.1        ||
=====================================================
+---------------------------------------------------+
|                TOPOLCO PREDICTION                 |
+---------------------------------------------------+
|                                                   |
| PREDICTOR VARIABLES (min)                         |
|                                                   |
| Max turbo frequency: {speed} GHz
| Basin cells: {int(cells)}
| RAM memory: {ram} Gb
| Cores: {n_core}
| Threads: {n_thr}
+---------------------------------------------------+
|                                                   |
| EXPECTED TIME (min)                               |
|                                                   |
| Expected mean time: {top_y_sim_mean:.3f} min
| Estimated time range: {top_y_sim_min:.3f} min  to {top_y_sim_max:.3f} min
+---------------------------------------------------+
|                                                   |
| MODEL VALIDATION ({n_iter} MC simulations)
|                                                   |
| Average RMSE: {top_mean_rmse:.3f}
| Average R²:   {top_mean_rsqr:.3f}
+---------------------------------------------------+

=====================================================

+---------------------------------------------------+
|        HYDROLOGICAL SIMULATION PREDICTION         |
+---------------------------------------------------+
|                                                   |
| PREDICTOR VARIABLES (min)                         |
|                                                   |
| Max turbo frequency: {speed} GHz
| Basin cells: {int(cells)}
| Time steps: {int(timesteps)}
| Input gauges: {n_in}
| Output gauges: {n_out}
+---------------------------------------------------+
|                                                   |
| EXPECTED TIME (min)                               |
|                                                   |
| Expected mean time: {sim_y_sim_mean:.3f} min
| Estimated time range: {sim_y_sim_min:.3f} min  to {sim_y_sim_max:.3f} min
+---------------------------------------------------+
|                                                   |
| MODEL VALIDATION ({n_iter} MC simulations)
|                                                   |
| Average RMSE: {sim_mean_rmse:.3f}
| Average R²:   {sim_mean_rsqr:.3f}
+---------------------------------------------------+
"""

# Print the generated prediction string to the console
print(prediction)

prediction_file = script_dir / "Prediction_results.txt"    # Define the path and filename for saving the results 

# Open the file in write mode ('w'). 'w' overwrites the file if it exists, and creates it otherwise.
# Specify encoding='utf-8' for broad compatibility with various characters.
with open(prediction_file, 'w', encoding='utf-8') as archivo:
    archivo.write(prediction) # Write the contents of the 'prediction' string to the file
    


#%%#######################################################################################################################
#################                             FINAL CODIGO                       #########################################
##########################################################################################################################

run_time = (time.time() - tic)
hours_ = run_time // 3600.0
minutes_ = round((run_time / 3600.0 - hours_) * 60.0, 1)
text_ = f'Execution total time was {hours_} hours and {minutes_} minutes'
len_text = len(text_)
len_print = len_text + 2 * 10
len_blank = (len_print - 2)
print(len_print * '#')
print('#' + len_blank * ' ' + '#')
print('#' + 9 * ' ' + text_ + 9 * ' ' + '#')
print('#' + len_blank * ' ' + '#')
print(len_print * '#')

input("\n--- Press Enter to close the window ---")