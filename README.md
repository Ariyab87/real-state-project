# Real Estate Data Analysis

## Overview
This project involves analyzing a real estate dataset using Python and various data science libraries. The tasks include data preprocessing, visualization, and machine learning models to predict house prices based on different features.

## Dataset
The dataset contains various attributes related to houses, such as price, square footage, number of floors, waterfront presence, and more.

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Steps and Code

### 1. Display the Data Types of Each Column
```python
import pandas as pd

# Load your dataset (replace 'file.csv' with the actual filename)
df = pd.read_csv('file.csv')

# Display data types
print(df.dtypes)
```

### 2. Drop "id" and "Unnamed: 0" Columns and Show Statistical Summary
```python
# Drop columns
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

# Display statistical summary
print(df.describe())
```

### 3. Count Unique Floor Values
```python
# Count unique floor values and convert to DataFrame
floor_counts = df['floors'].value_counts().to_frame()
print(floor_counts)
```

### 4. Boxplot for Waterfront vs. Price Outliers
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="waterfront", y="price", data=df)

# Show plot
plt.show()
```

### 5. Regression Plot for sqft_above vs. Price Correlation
```python
# Create regression plot
plt.figure(figsize=(8, 6))
sns.regplot(x="sqft_above", y="price", data=df)

# Show plot
plt.show()
```

### 6. Fit Linear Regression Model for sqft_living
```python
from sklearn.linear_model import LinearRegression

# Define X and y
X = df[["sqft_living"]]
y = df["price"]

# Fit model
model = LinearRegression()
model.fit(X, y)

# Calculate R^2
r2 = model.score(X, y)
print(f"R^2: {r2}")
```

### 7. Fit Linear Regression Model Using Multiple Features
```python
# Define features
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]
y = df["price"]

# Fit model
model = LinearRegression()
model.fit(X, y)

# Calculate R^2
r2 = model.score(X, y)
print(f"R^2: {r2}")
```

### 8. Create a Pipeline with Scaling, Polynomial Transform, and Linear Regression
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])

# Fit pipeline
pipeline.fit(X, y)

# Calculate R^2
r2 = pipeline.score(X, y)
print(f"R^2: {r2}")
```

### 9. Ridge Regression (Regularization = 0.1)
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Ridge model
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)

# Calculate R^2 on test data
r2 = ridge.score(X_test, y_test)
print(f"R^2: {r2}")
```

### 10. Perform Second-Order Polynomial Transform and Fit Ridge Regression
```python
# Apply polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit Ridge model
ridge_poly = Ridge(alpha=0.1)
ridge_poly.fit(X_train_poly, y_train)

# Calculate R^2 on test data
r2 = ridge_poly.score(X_test_poly, y_test)
print(f"R^2: {r2}")
```

## Conclusion
This project demonstrates various data analysis techniques, visualization methods, and machine learning models to analyze and predict house prices.

## License
This project is for educational purposes only. Feel free to modify and use it for learning!

