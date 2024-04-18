import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('https://raw.githubusercontent.com/SridharModukuru/Onsite-Health-Diagnostic-OHD-circleci/main/data.csv')

df['sqft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_basement']
df['yr_built'] = np.where(df['yr_renovated'] != 0, df['yr_renovated'], df['yr_built'])
df.drop(columns=['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'view', 'yr_renovated', 'date', 'floors', 'waterfront', 'street', 'statezip', 'country'], inplace=True)

label_encoder = LabelEncoder()
df['city'] = label_encoder.fit_transform(df['city'])

y = df['price'].copy()
df.drop(columns=['price'], inplace=True)

scaler = StandardScaler()
# Reshape the input data from 1D to 2D before applying fit_transform
df['sqft'] = scaler.fit_transform(df['sqft'].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

file_name = 'price_model.sav'
joblib.dump(lr, file_name)

loaded_model = joblib.load(file_name)

# Reshape the test data before making predictions
X_test['sqft'] = scaler.transform(X_test['sqft'].values.reshape(-1, 1))

pred = loaded_model.predict(X_test)
