import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\HP\OneDrive\Machine_Learning_projects\bengaluru_house_predictor\Bengaluru_House_Data.csv")

# Initial Data Inspection
print("Initial Data Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Feature Selection
df = df.drop(columns=['area_type', 'availability', 'society', 'balcony'])

# Handle Missing Values
df = df.dropna(subset=['size', 'price'])  # Critical features
df['location'] = df['location'].fillna('Sarjapur Road')
df['bath'] = df['bath'].fillna(df['bath'].median())  # Using median instead of mode

# Remove Duplicates
df = df.drop_duplicates()

# Feature Engineering
# Convert size to numeric (extract bedroom count)
df['size'] = df['size'].str.split(' ').str[0].astype(int)

# Convert total_sqft to numeric (handle ranges)
def convert_range_to_mean(x):
    if isinstance(x, str):
        if '-' in x:
            low, high = x.split('-')
            return (float(low) + float(high)) / 2
        elif x.lower() in ['nan', 'na']:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_range_to_mean)
df = df.dropna(subset=['total_sqft'])

# Create price per sqft feature
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['total_sqft', 'bath', 'price', 'price_per_sqft']:
    df = remove_outliers(df, col)

print("\nData Shape After Cleaning:", df.shape)

# EDA Visualizations
plt.figure(figsize=(15, 10))
sns.pairplot(df[['total_sqft', 'size', 'bath', 'price']])
plt.suptitle('Feature Relationships', y=1.02)
plt.show()

# Prepare data for modeling
X = df.drop(columns=['price', 'price_per_sqft'])
y = df['price']

# Split data before any preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define preprocessing
numeric_features = ['total_sqft', 'size', 'bath']
categorical_features = ['location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nModel Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Feature Importance (after one-hot encoding)
feature_names = numeric_features + list(
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

importances = model.named_steps['regressor'].feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Save cleaned data and model
df.to_csv("cleaned_house_data.csv", index=False)

# For saving the model (uncomment if needed)
# import joblib
# joblib.dump(model, 'house_price_model.pkl')