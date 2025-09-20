import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Machine_Learning_projects\SMS_SPAM_DETECTOR\u.csv")

# Print dataset snippet (Annexure: Dataset Sample)
print("\n===== Annexure C – Sample Dataset (Car Price Data) =====\n")
print(df.head(10))

# Visualization 1: Distribution of Car Prices
plt.figure(figsize=(8,5))
sns.histplot(df['Present_Price'], bins=10, kde=True)
plt.title("Distribution of Car Prices", fontsize=14)
plt.xlabel("Car Price (in Lakhs)")
plt.ylabel("Count")
plt.show()

# Visualization 2: Price vs Year
plt.figure(figsize=(8,5))
sns.scatterplot(x="Year", y="Present_Price", data=df, hue="Fuel_Type", style="Transmission", s=100)
plt.title("Car Price vs Year of Manufacture", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Car Price (in Lakhs)")
plt.show()

# Visualization 3: Fuel Type vs Average Price
plt.figure(figsize=(7,5))
sns.barplot(x="Fuel_Type", y="Present_Price", data=df, estimator='mean')
plt.title("Average Car Price by Fuel Type", fontsize=14)
plt.ylabel("Average Price (Lakhs)")
plt.show()

# Visualization 4: Transmission Effect on Price
plt.figure(figsize=(6,5))
sns.boxplot(x="Transmission", y="Present_Price", data=df)
plt.title("Car Price Distribution by Transmission Type", fontsize=14)
plt.show()

# Annexure Heading
print("\n===== Annexure D – Graphical Representations (Car Price Prediction Project) =====\n")
print("1. Distribution of Car Prices")
print("2. Car Price vs Year (with Fuel Type & Transmission)")
print("3. Average Car Price by Fuel Type")
print("4. Car Price Distribution by Transmission Type")
