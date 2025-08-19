# ================================
# Level 1 - Cognifyz Internship
# Data Exploration & Analysis
# ================================

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset into a pandas DataFrame
# r"" is used to handle the Windows file path properly
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Dataset.csv")

# Replace missing values in the 'Cuisines' column with "Unknown"
df['Cuisines'] = df['Cuisines'].fillna("Unknown")

# -------- Task 1: Data Exploration --------

# Print shape of dataset (rows, columns)
print("Shape:", df.shape)

# Print count of missing values in each column
print("Missing values:\n", df.isnull().sum())

# Count how many restaurants have a rating of 0
print("Zero ratings:", (df['Aggregate rating'] == 0).sum())

# Check correlation between Latitude and Rating
print("Correlation (Lat, Rating):", df['Latitude'].corr(df['Aggregate rating']))

# Check correlation between Longitude and Rating
print("Correlation (Lon, Rating):", df['Longitude'].corr(df['Aggregate rating']))

# -------- Task 2: Descriptive Analysis --------

print("\n--- Top Cities ---")
# Show the top 10 cities with most restaurants
print(df['City'].value_counts().head(10))

# Split the 'Cuisines' column (comma-separated) into individual cuisines
# explode() creates a new row for each cuisine
cuisine_exploded = df.assign(Cuisine=df['Cuisines'].str.split(',')).explode('Cuisine')

# Remove extra spaces around cuisine names
cuisine_exploded['Cuisine'] = cuisine_exploded['Cuisine'].str.strip()

print("\n--- Top Cuisines ---")
# Show the top 10 most common cuisines
print(cuisine_exploded['Cuisine'].value_counts().head(10))

# -------- Task 3: Geospatial Analysis --------

# Scatter plot of restaurant locations using Latitude & Longitude
plt.scatter(df['Longitude'], df['Latitude'], s=1)  # s=1 makes points small
plt.title('Restaurant Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
