# ================================
# Level 3 - Cognifyz Internship
# Predictive Modeling & Preferences
# ================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Dataset.csv")

# Fill missing cuisines with "Unknown"
df['Cuisines'] = df['Cuisines'].fillna("Unknown")

# Convert categorical yes/no values into binary (0/1)
df['has_table_booking'] = (df['Has Table booking'].str.lower() == 'yes').astype(int)
df['has_online_delivery'] = (df['Has Online delivery'].str.lower() == 'yes').astype(int)
df['is_delivering_now'] = (df['Is delivering now'].str.lower() == 'yes').astype(int)

# Create some engineered features
df['name_len'] = df['Restaurant Name'].str.len()         # length of restaurant name
df['address_len'] = df['Address'].str.len()              # length of address
df['cuisines_count'] = df['Cuisines'].str.split(',').apply(len)  # number of cuisines offered

# -------- Task 1: Predictive Modeling --------

# Select features (independent variables)
features = ['Average Cost for two','Price range','Votes',
            'has_table_booking','has_online_delivery','is_delivering_now',
            'cuisines_count','name_len','address_len']

X = df[features]                     # features
y = df['Aggregate rating']           # target variable (what we predict)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, min_samples_leaf=5),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=200)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)                       # train model
    preds = model.predict(X_test)                     # make predictions
    rmse = (mean_squared_error(y_test, preds)) ** 0.5 # Root Mean Squared Error (fixed)
    mae = mean_absolute_error(y_test, preds)          # Mean Absolute Error
    r2 = r2_score(y_test, preds)                      # R² Score (goodness of fit)
    print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# -------- Task 2: Customer Preference --------

# Expand cuisines column so each row has one cuisine (instead of comma-separated list)
cuisine_exploded = df.assign(Cuisine=df['Cuisines'].str.split(',')).explode('Cuisine')

# Clean up extra spaces in cuisine names
cuisine_exploded['Cuisine'] = cuisine_exploded['Cuisine'].str.strip()

# Top cuisines by total votes
print("\nTop cuisines by votes:\n", 
      cuisine_exploded.groupby('Cuisine')['Votes'].sum().sort_values(ascending=False).head(10))

# Compute stats per cuisine: number of unique restaurants + average rating
cuisine_stats = (cuisine_exploded.groupby('Cuisine')
                 .agg(Restaurants=('Restaurant ID','nunique'),
                      Avg_Rating=('Aggregate rating','mean'))
                 .reset_index())

# Top cuisines by rating (only considering cuisines available in >= 30 restaurants)
print("\nTop cuisines by rating (>=30 restaurants):\n", 
      cuisine_stats[cuisine_stats['Restaurants']>=30]
      .sort_values('Avg_Rating',ascending=False).head(10))

# -------- Task 3: Visualizations --------

# Histogram of restaurant ratings
plt.hist(df['Aggregate rating'], bins=20, color='skyblue')
plt.title('Distribution of Ratings')
plt.show()

# Average rating for top 10 most common cuisines
top10_cuisines = cuisine_exploded['Cuisine'].value_counts().head(10).index
avg_rating_cuisines = (cuisine_exploded[cuisine_exploded['Cuisine'].isin(top10_cuisines)]
                       .groupby('Cuisine')['Aggregate rating'].mean())
avg_rating_cuisines.plot(kind='bar', title='Avg Rating by Top 10 Cuisines')
plt.show()

# Average rating for restaurants in top 10 cities (by restaurant count)
top10_cities = df['City'].value_counts().head(10).index
avg_rating_cities = df[df['City'].isin(top10_cities)].groupby('City')['Aggregate rating'].mean()
avg_rating_cities.plot(kind='bar', title='Avg Rating by Top 10 Cities')
plt.show()
