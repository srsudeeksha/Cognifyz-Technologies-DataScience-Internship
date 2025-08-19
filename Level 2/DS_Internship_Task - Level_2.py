# ================================
# Level 2 - Cognifyz Internship
# Business Insights
# ================================

import pandas as pd

# Load dataset into pandas DataFrame
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Dataset.csv")

# Replace missing values in 'Cuisines' with "Unknown"
df['Cuisines'] = df['Cuisines'].fillna("Unknown")

# Convert categorical yes/no columns to binary (0/1)
# If "Yes" → 1, else 0
df['has_table_booking'] = (df['Has Table booking'].str.lower() == 'yes').astype(int)
df['has_online_delivery'] = (df['Has Online delivery'].str.lower() == 'yes').astype(int)

# -------- Task 1: Table Booking & Online Delivery --------

# % of restaurants that provide table booking
print("\n% with Table Booking:", df['has_table_booking'].mean()*100)

# % of restaurants that provide online delivery
print("% with Online Delivery:", df['has_online_delivery'].mean()*100)

# Compare average rating of restaurants with and without table booking
print("Avg Rating by Booking:\n", df.groupby('has_table_booking')['Aggregate rating'].mean())

# Compare average rating of restaurants with and without online delivery
print("Avg Rating by Online Delivery:\n", df.groupby('has_online_delivery')['Aggregate rating'].mean())

# Compare % of restaurants with delivery across different price ranges
print("Delivery % by Price Range:\n", df.groupby('Price range')['has_online_delivery'].mean()*100)

# -------- Task 2: Price Range Analysis --------

# Find the most common price range (mode of the column)
print("\nMost common price range:", df['Price range'].value_counts().idxmax())

# Compare average rating of restaurants across price ranges
print("Avg Rating by Price Range:\n", df.groupby('Price range')['Aggregate rating'].mean())

# Compare average rating by rating color (like 'Green', 'Orange', etc.)
# Sorting in descending order to see the highest rated color first
print("Avg Rating by Color:\n", df.groupby('Rating color')['Aggregate rating'].mean().sort_values(ascending=False))

# -------- Task 3: Feature Engineering --------

# Create new features for analysis
df['name_len'] = df['Restaurant Name'].str.len()       # length of restaurant name
df['address_len'] = df['Address'].str.len()            # length of address
df['cuisines_count'] = df['Cuisines'].str.split(',').apply(len)  # number of cuisines per restaurant

# Preview new engineered features along with booking & delivery indicators
print("\nFeature Engineering Preview:\n", 
      df[['Restaurant Name','name_len','address_len','cuisines_count','has_table_booking','has_online_delivery']].head())
