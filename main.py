import pandas as pd
import numpy as np

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?")

print(df.head())

obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())

# Approach #1 - Find and Replace
print("Find and replace")
obj_df["num_cylinders"].value_counts()
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}

obj_df = obj_df.replace(cleanup_nums)
print(obj_df.head())

# Approach #2 - Label Encoding
print("Label encoding")
obj_df["body_style"] = obj_df["body_style"].astype('category')

obj_df["body_style_cat"] = obj_df["body_style"].cat.codes
print(obj_df.head())

# Approach #3 - One-Hot encoding
print("One-Hot encoding")
print(pd.get_dummies(obj_df, columns=["drive_wheels"]).head())
print(pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head())

# Approach #4 - Custom Binary Encoding
print("Custom binary encoding")
obj_df["engine_type"].value_counts()

obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)

print(obj_df[["make", "engine_type", "OHC_Code"]].head())


