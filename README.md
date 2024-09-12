# Mobher Material predicton System(beta)

## Overview

This project provides an AI-based system for predicting the required building materials and associated design specifications based on the environment and building elements. It utilizes machine learning models to predict quantities of various materials and recommend the best cement type for different building elements.

## Project Structure

1. **Data Preparation**: Clean and merge datasets.
2. **Feature Engineering**: Encode categorical variables and scale features.
3. **Model Training**: Train Random Forest models for both regression (numeric targets) and classification (categorical targets).
4. **Prediction and Evaluation**: Generate predictions and evaluate model performance.
5. **Utility Functions**: Provide functions to generate building material descriptions based on the governorate.

## Files

- `governorates.csv`: Contains the mapping of governorates to their environmental conditions.
- `Concrete.csv`: Includes data on concrete mixtures and their environmental compatibility.
- `materials_dataset.csv`: Contains information on building materials including quantity requirements.
- `height_conditions_egyptian_code.csv`: Lists height and zoning conditions according to Egyptian code.

## How to run our project on your device locally? 

Make sure you have: [ Python , Git ] <br>
First, go to the folder where you want to download the project, then open CMD inside it and type the following codes:
```Cmd
git clone https://github.com/Mobher-team/ML-work.git
```
Then after it's done Write this to install all requerments:
```Cmd
cd ML-work
```
```Cmd
python -m pip install -r requirements.txt
```
For starting the API Write this:
```Cmd
uvicorn main:app --host 127.0.0.1 --port 8000
```
And then enter this link to see the Local functional API Documentation: http://127.0.0.1:8000/docs <br>
To test the API click on 'Try it out' Then replace 'String' By any Egyption governorate and Make sure that the first letter is capital. <br>
There is a link for non-functional API Documentationo: https://youssefabukhalifa.github.io/Api-Documentation-Mobher-/

## Code Description

### Importing Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')
```

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **sklearn**: Machine learning tools (preprocessing, model training, evaluation).
- **joblib**: Model saving and loading.

### Data Loading and Preparation

```python
governorates_df = pd.read_csv('governorates.csv')
concrete_df = pd.read_csv('Concrete.csv')
materials_df = pd.read_csv('materials_dataset.csv')
height_conditions_df = pd.read_csv('height_conditions_egyptian_code.csv')

# Clean and prepare data
materials_df.rename(columns={'Quantity for 1m²': 'Quantity_for_1m²'}, inplace=True)
concrete_df['Environment'] = concrete_df['Environment'].str.strip()
concrete_df['Building Element'] = concrete_df['Building Element'].str.strip()
```

- **Loading Datasets**: Read CSV files into DataFrames.
- **Data Cleaning**: Strip whitespace and rename columns for consistency.

### Merging Datasets

```python
data = pd.merge(concrete_df, materials_df[['Environments', 'Material', 'Type', 'Quantity_for_1m²']],
                left_on="Environment", right_on="Environments", how="left")
```

- **Merge**: Combine datasets based on environment.

### Encoding and Scaling

```python
# Initialize encoders and scalers
encoders = {
    'Environment': LabelEncoder(),
    'Building Element': LabelEncoder(),
    'Best Cement': LabelEncoder(),
    'Material': LabelEncoder(),
    'Type': LabelEncoder()
}

for column, encoder in encoders.items():
    if column in data.columns:
        data[column] = encoder.fit_transform(data[column])

scaler = StandardScaler()

# Scale features and numeric targets
X = data[features]
y_numeric = data[numeric_targets]
y_categorical = data[categorical_targets]

X_scaled = scaler.fit_transform(X)
y_numeric_scaled = StandardScaler().fit_transform(y_numeric)
```

- **Encoding**: Convert categorical variables to numerical values.
- **Scaling**: Standardize features and targets.

### Model Training

```python
# Split the data
X_train, X_test, y_numeric_train, y_numeric_test, y_categorical_train, y_categorical_test = train_test_split(
    X_scaled, y_numeric_scaled, y_categorical, test_size=0.2, random_state=42)

# Train Random Forest models
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_numeric_train)

rf_classifiers = {}
for target in categorical_targets:
    rf_classifiers[target] = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifiers[target].fit(X_train, y_categorical_train[target])
```

- **Training**: Fit Random Forest models for regression and classification.

### Model Evaluation

```python
# Evaluate models
numeric_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_numeric_test, numeric_pred)
print(f"Mean Squared Error for numeric predictions: {mse}")

for target in categorical_targets:
    categorical_pred = rf_classifiers[target].predict(X_test)
    accuracy = accuracy_score(y_categorical_test[target], categorical_pred)
    print(f"Accuracy for {target} predictions: {accuracy}")
```

- **Evaluation**: Calculate mean squared error for regression and accuracy for classification.

### Saving Models and Scalers

```python
# Save models
joblib.dump(rf_regressor, 'rf_regressor.joblib')
for target, model in rf_classifiers.items():
    joblib.dump(model, f'rf_classifier_{target}.joblib')

# Save scaler
joblib.dump(scaler, 'feature_scaler.joblib')
```

- **Saving**: Store trained models and scalers for future use.

### Utility Function: Generating Building Descriptions

```python
def generate_building_description(governorate):
    try:
        if governorate not in gov_env_map:
            return None, None, f"Error: '{governorate}' is not a valid governorate. Please choose from: {', '.join(gov_env_map.keys())}"

        environment = gov_env_map[governorate]

        # Encode environment
        env_encoded = encoders['Environment'].transform([environment])[0]

        # List of building elements
        building_elements = ['Foundation', 'Slabs (Floors)', 'Beams and Columns', 'Exterior Walls', 'Roof']
        result_data = []

        for building_element in building_elements:
            try:
                # Encode building element
                be_encoded = encoders['Building Element'].transform([building_element])[0]

                # Prepare input data for the model
                input_data = scaler.transform([[env_encoded, be_encoded]])

                # Get numeric predictions
                numeric_pred = rf_regressor.predict(input_data)
                numeric_pred = StandardScaler().fit(y_numeric).inverse_transform(numeric_pred)

                # Get categorical predictions
                categorical_pred = {}
                for target, model in rf_classifiers.items():
                    pred = model.predict(input_data)[0]
                    categorical_pred[target] = encoders[target].inverse_transform([pred])[0]

                # Append the prediction results for this building element
                result_data.append({
                    'Building Element': building_element,
                    'Best Cement': categorical_pred['Best Cement'],
                    'Cement (kg/m³)': max(0, numeric_pred[0][0]),
                    'Sand (kg/m³)': max(0, numeric_pred[0][1]),
                    'Aggregates (kg/m³)': max(0, numeric_pred[0][2]),
                    'Water (liters/m³)': max(0, numeric_pred[0][3]),

                })

            except Exception as e:
                result_data.append({
                    'Building Element': building_element,
                    'Error': f"Error in prediction: {str(e)}"
                })

        # Convert the result data into a DataFrame
        result_df = pd.DataFrame(result_data)

        # Add height conditions information
        height_info = height_conditions_df[height_conditions_df['Zoning Area'] == environment]

        # Check if height_info is empty
        if height_info.empty:
            height_info_dict = {
                'Error': f"No height condition data found for environment '{environment}'"
            }
        else:
            height_info_row = height_info.iloc[0]
            height_info_dict = {
                'Max Height (m)': height_info_row['Max Height (m)'],
                'Min Height (m)': height_info_row['Min Height (m)'],
                'Floor Area Ratio (FAR)': height_info_row['Floor Area Ratio (FAR)'],
                'Setback Requirements': height_info_row['Setback Requirements'],
                'Usage Restrictions': height_info_row['Usage Restrictions'],
                'Height Condition': height_info_row['Height Condition'],
                'Code Reference': height_info_row['Code Reference']
            }

        # Add materials information
        materials_info = materials_df[materials_df['Environments'] == environment]
        if materials_info.empty:
            materials_info = pd.DataFrame({'Error': [f"No materials data found for environment '{environment}'"]})
        else:
            materials_info = materials_info[['Material', 'Type', 'Quantity_for_1m²']]

        return result_df, height_info_dict, materials_info

    except Exception as e:
        return None, None, f"Error in prediction: {str(e)}"
```

- **generate_building_description**: Generates a detailed description of the building requirements based on the provided governorate.

## Example Usage

```python
description_df, height_info, materials_info = generate_building_description('Damietta')

print("Building Element Predictions:")
print(description_df)
print("\nHeight and Zoning Information:")
for key, value in height_info.items():
    print(f"{key}: {value}")
print("\nMaterials Information:")
print(materials_info)
```

## Requirements

- pandas
- numpy
- scikit-learn
- joblib

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn joblib
```

## License

This project belongs to Mobher Team.
