from sklearn.model_selection import train_test_split

# Suppose merged_df is your final preprocessed dataframe

import pandas as pd

# Example: Load merged_df from a CSV file (update the path as needed)
merged_df = pd.read_csv("D:/project 7th semester/processed_data/preprocessed_olist_data_strict.csv")

# Optionally, drop non-feature columns (like IDs or text if not using them)
X = merged_df.drop(columns=["customer_id", "product_id", "order_id", "review_id"])
y = None  # For product recommendation, usually target is implicit (like interaction, purchase, rating)
# If you have a target column, assign it here, e.g., y = merged_df['purchase_flag']

# For implicit feedback recommendation (like browsing/purchase history):
# Usually we split by users to avoid data leakage

train_df, test_df = train_test_split(
    merged_df, 
    test_size=0.2,  # 20% for testing
    random_state=42,
    stratify=None  # or stratify by customer_id if needed
)

# Save to CSV if needed
train_df.to_csv("processed_data/train_data.csv", index=False)
test_df.to_csv("processed_data/test_data.csv", index=False)

print(f"Training set: {train_df.shape}")
print(f"Testing set: {test_df.shape}")
