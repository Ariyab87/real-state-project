Data Analysis with Python - Coursera Project
This repository contains a Python script for performing data analysis on a synthetic real estate dataset. The script includes data generation, data cleaning, exploratory data analysis (EDA), and predictive modeling using linear regression and ridge regression.

Table of Contents
Project Overview

Installation

Usage

Data Generation

Exploratory Data Analysis (EDA)

Predictive Modeling

Results

License

Project Overview
This project demonstrates the use of Python for data analysis and predictive modeling. The dataset used is synthetically generated and includes features such as price, square footage, number of bedrooms, and more. The analysis includes data cleaning, visualization, and building predictive models to estimate house prices.

Installation
To run this project, you need to have Python installed along with the following libraries:

pandas

numpy

seaborn

matplotlib

scikit-learn

You can install these libraries using pip:

bash
Copy
pip install pandas numpy seaborn matplotlib scikit-learn
Usage
Clone the repository:

bash
Copy
git clone https://github.com/your-username/data-analysis-with-python-coursera.git
Navigate to the project directory:

bash
Copy
cd data-analysis-with-python-coursera
Run the Python script:

bash
Copy
python data_analyses_with_python_coursera_.py
Data Generation
The script generates a synthetic dataset with 1000 rows and the following columns:

id: Unique identifier for each row.

Unnamed: 0: Another unique identifier.

price: Price of the house.

sqft_living: Square footage of the living area.

sqft_above: Square footage above ground.

floors: Number of floors.

waterfront: Whether the house has a waterfront view (0 or 1).

lat: Latitude of the house location.

bedrooms: Number of bedrooms.

sqft_basement: Square footage of the basement.

view: Quality of the view (0 to 4).

bathrooms: Number of bathrooms.

sqft_living15: Average square footage of living area for 15 nearest neighbors.

grade: Overall grade of the house (1 to 12).

The dataset is saved as real_estate_data.csv.

Exploratory Data Analysis (EDA)
The script performs the following EDA tasks:

Data Cleaning: Drops unnecessary columns (id and Unnamed: 0).

Statistical Summary: Displays summary statistics for the dataset.

Value Counts: Counts the number of houses with different numbers of floors.

Visualization:

Boxplot of house prices based on waterfront view.

Scatter plot with regression line for square footage above ground vs. price.

Predictive Modeling
The script builds and evaluates several predictive models:

Simple Linear Regression: Predicts house prices based on square footage of the living area.

Multiple Linear Regression: Predicts house prices using multiple features.

Polynomial Regression: Uses polynomial features to improve the model.

Ridge Regression: Regularized linear regression to prevent overfitting.

The R² score is used to evaluate the performance of each model.

Results
The R² scores for the models are as follows:

Simple Linear Regression: R² = [value]

Multiple Linear Regression: R² = [value]

Polynomial Regression: R² = [value]

Ridge Regression: R² = [value]

License
This project is licensed under the MIT License. See the LICENSE file for details.
