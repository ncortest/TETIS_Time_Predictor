# 🚀 TETIS\_Time\_Predictor: Computational Performance Prediction

This repository hosts the **trained Random Forest (RF) predictive tool** designed to estimate the execution time of the **TETIS v9.1** distributed hydrological model.

The tool allows researchers and end-users to forecast runtime based on their specific simulation configurations and hardware.

## 📦 Repository Structure

| Directory | Content | Description |
| :--- | :--- | :--- |
| **`TETIS_time_Predictor.zip`** | Compressed Tool | Contains the core Python scripts, the trained RF models, and the necessary input template (`User_predictor_variables.csv`). |
| **`LICENSE`** | License File | Software license details. |

-----

## 🛠️ Performance Prediction Tool Usage

To predict TETIS execution times for your specific simulation and hardware configurations, follow these simple steps:

### Step 1: Download and Prepare the Tool

1.  **Download the folder** `TETIS_Time_Predictor.zip` to your local machine.
2.  **Uncompress the Predictor .zip**. This action will unpack the necessary execution files and the input template.

### Step 2: Define Your Scenario

1.  **Locate Input File:** Navigate into the uncompressed directory and find the input template: **`User_predictor_variables.csv`**.
2.  **Modify Predictor Data:** Open **`User_predictor_variables.csv`** and modify the values of the predictor variables  (e.g., Max turbo frequency, Basin area_km2, Cell size_m, RAM, Initial date, Final date, Delta t_minutes, input gauges, output gauges, RAM memory, Cores, Threads) to match **your specific simulation scenarios and hardware configuration**.
   
**Note:** the tool has been programmed to predict combinations of parameters provided by the user. It should be noted that some combinations do not generate results, therefore it is advisable to enter all the parameters.

### Step 3: Run the Prediction

1.  **Execute the 'RF_Prediction_TETIS.py'** to initiate the prediction process. (You will need Python and the necessary ML libraries installed, such as `scikit-learn` and `pandas`).
2.  **View Results in output file `Prediction_results.txt`**. This file contains the estimated execution times (runtime) and the associated uncertainty ranges for the scenarios defined in your input CSV file.

-----

## 🔗 Citation

Please cite the associated publication when using this predictive tool or its underlying models:

\[Insert Full Article Title Here]

**DOI:** \[Insert Article DOI Here]
