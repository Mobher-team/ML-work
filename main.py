import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI()

# Load datasets
governorates_df = pd.read_csv('governorates.csv')
concrete_df = pd.read_csv('Concrete.csv')
materials_df = pd.read_csv('materials_dataset.csv')
height_conditions_df = pd.read_csv('height_conditions_egyptian_code.csv')

# Clean and prepare data
materials_df.rename(columns={'Quantity for 1m²': 'Quantity_for_1m²'}, inplace=True)
concrete_df['Environment'] = concrete_df['Environment'].str.strip()
concrete_df['Building Element'] = concrete_df['Building Element'].str.strip()

# Merge datasets
data = pd.merge(concrete_df, materials_df[['Environments', 'Material', 'Type', 'Quantity_for_1m²']], 
                left_on="Environment", right_on="Environments", how="left")

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

# Select features and targets
features = ['Environment', 'Building Element']
numeric_targets = ['Cement (kg/m³)', 'Sand (kg/m³)', 'Aggregates (kg/m³)', 'Water (liters/m³)', 'Quantity_for_1m²']
categorical_targets = ['Best Cement', 'Material', 'Type']

X = data[features]
y_numeric = data[numeric_targets]
y_categorical = data[categorical_targets]

# Scale features and numeric targets
X_scaled = scaler.fit_transform(X)
y_numeric_scaled = StandardScaler().fit_transform(y_numeric)

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

# Evaluate models
numeric_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_numeric_test, numeric_pred)
print(f"Mean Squared Error for numeric predictions: {mse}")

for target in categorical_targets:
    categorical_pred = rf_classifiers[target].predict(X_test)
    accuracy = accuracy_score(y_categorical_test[target], categorical_pred)
    print(f"Accuracy for {target} predictions: {accuracy}")

# Save models
joblib.dump(rf_regressor, 'rf_regressor.joblib')
for target, model in rf_classifiers.items():
    joblib.dump(model, f'rf_classifier_{target}.joblib')

# Save scaler
joblib.dump(scaler, 'feature_scaler.joblib')

# Create governorate to environment mapping
gov_env_map = dict(zip(governorates_df['Governorate'], governorates_df['Environment']))

# Request body model for FastAPI
class GovernorateRequest(BaseModel):
    governorate: str

# API route to generate building description
@app.post("/generate_description/")
def generate_building_description(request: GovernorateRequest):
    governorate = request.governorate
    if governorate not in gov_env_map:
        raise HTTPException(status_code=400, detail=f"'{governorate}' is not a valid governorate.")

    environment = gov_env_map[governorate]

    # Encode environment
    env_encoded = encoders['Environment'].transform([environment])[0]

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
                'Cement (kg/m³)': float(max(0, numeric_pred[0][0])),
                'Sand (kg/m³)': float(max(0, numeric_pred[0][1])),
                'Aggregates (kg/m³)': float(max(0, numeric_pred[0][2])),
                'Water (liters/m³)': float(max(0, numeric_pred[0][3])),
        
            })

        except Exception as e:
            result_data.append({
                'Building Element': building_element,
                'Error': f"Error in prediction: {str(e)}"
            })

    # Add height conditions information
    height_info = height_conditions_df[height_conditions_df['Zoning Area'] == environment]
    if height_info.empty:
        height_info_dict = {'Error': f"No height condition data found for environment '{environment}'"}
    else:
        height_info_row = height_info.iloc[0]
        height_info_dict = {
            'Max Height (m)': float(height_info_row['Max Height (m)']),
            'Min Height (m)': float(height_info_row['Min Height (m)']),
            'Floor Area Ratio (FAR)': float(height_info_row['Floor Area Ratio (FAR)']),
            'Setback Requirements': height_info_row['Setback Requirements'],
            'Usage Restrictions': height_info_row['Usage Restrictions'],
            'Height Condition': height_info_row['Height Condition'],
            'Code Reference': height_info_row['Code Reference']
        }

    # Add materials information
    materials_info = materials_df[materials_df['Environments'] == environment]
    if materials_info.empty:
        materials_info = [{'Error': f"No materials data found for environment '{environment}'"}]
    else:
        materials_info = materials_info[['Material', 'Type', 'Quantity_for_1m²']].to_dict(orient='records')

    return {
        "description": result_data,
        "height_info": height_info_dict,
        "materials_info": materials_info
    }

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)