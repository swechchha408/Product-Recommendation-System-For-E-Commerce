import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your final merged dataset
df = pd.read_csv("D:/project 7th semester/processed_data/preprocessed_olist_data_strict.csv")  # Change file name if needed

# Ensure required columns exist
df = df[['customer_id', 'product_id', 'order_id', 'review_score']]

# Create folder if not exists
os.makedirs("data", exist_ok=True)

# ✅ Step 1: Create Interaction Table
interaction_table = df.groupby(['customer_id', 'product_id']).agg(
    interactions=('order_id', 'count'),
    avg_review_score=('review_score', 'mean')
).reset_index()

interaction_table['avg_review_score'].fillna(3, inplace=True)

# Save interaction table inside data folder
interaction_table.to_csv("data/interaction_table.csv", index=False)
print("✅ interaction_table.csv saved in data folder")

# ==============================

import json
from sklearn.preprocessing import MinMaxScaler

# Load interaction table
interaction_table = pd.read_csv("data/interaction_table.csv")

print("🔹 Before processing:")
print(interaction_table.head())

# ✅ Encode customer_id & product_id
interaction_table['customer_id_encoded'] = (
    interaction_table['customer_id'].astype('category').cat.codes
)
interaction_table['product_id_encoded'] = (
    interaction_table['product_id'].astype('category').cat.codes
)

# ✅ Save mappings for decoding later
customer_encoder = dict(zip(interaction_table['customer_id'], interaction_table['customer_id_encoded']))
product_encoder = dict(zip(interaction_table['product_id'], interaction_table['product_id_encoded']))

with open("data/customer_encoder.json", "w") as f:
    json.dump(customer_encoder, f)

with open("data/product_encoder.json", "w") as f:
    json.dump(product_encoder, f)

print("✅ Encoded customer & product IDs saved")

# ✅ Normalize ratings (0-1 scale)
scaler = MinMaxScaler(feature_range=(0, 1))
interaction_table['rating_normalized'] = scaler.fit_transform(
    interaction_table[['avg_review_score']]
)

print("✅ Ratings normalized")

# ✅ Drop original ID and raw rating columns
interaction_table.drop(
    ['customer_id', 'product_id', 'avg_review_score'],
    axis=1,
    inplace=True
)

print("\n✅ Final Training Data Columns:", interaction_table.columns.tolist())

# ✅ Save final encoded table
interaction_table.to_csv("data/interaction_table_encoded_train.csv", index=False)

print("\n🚀 Done! interaction_table_encoded_train.csv saved")
print(interaction_table.head())