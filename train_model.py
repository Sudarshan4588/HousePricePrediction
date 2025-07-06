import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('House_Price.csv')

# Select features and target
X = df[['number of bedrooms', 'number of bathrooms', 'living area',
        'lot area', 'number of floors', 'condition of the house',
        'grade of the house', 'Distance from the airport']]
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and feature names
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(list(X.columns), 'features.pkl')

print("✅ Model and feature list saved!")
