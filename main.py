# main.py
import os
import pandas as pd
from src.data_preprocessing import (
    preprocess_customers, preprocess_orders, preprocess_order_items,
    preprocess_products, preprocess_categories, preprocess_reviews,
    merge_all, clean_outliers_with_boxplots, feature_engineering, add_rfm_features,
    final_preprocessing, verify_dataset
)

# ------------------------
# 1. Load datasets
# ------------------------
customers = pd.read_csv("D:/project 7th semester/project 7th sem data/olist_customers_dataset.csv")
orders = pd.read_csv("D:/project 7th semester/project 7th sem data/olist_orders_dataset.csv")
order_items = pd.read_csv("D:/project 7th semester/project 7th sem data/olist_order_items_dataset.csv")
products = pd.read_csv("D:/project 7th semester/project 7th sem data/olist_products_dataset.csv")
categories = pd.read_csv("D:/project 7th semester/project 7th sem data/product_category_name_translation.csv")
reviews = pd.read_csv("D:/project 7th semester/project 7th sem data/olist_order_reviews_dataset.csv")

# ------------------------
# 2. Preprocess each dataset
# ------------------------
customers = preprocess_customers(customers)
orders = preprocess_orders(orders)
order_items = preprocess_order_items(order_items)
products = preprocess_products(products)
categories = preprocess_categories(categories)
reviews = preprocess_reviews(reviews)

# ------------------------
# 3. Merge datasets
# ------------------------
merged_df = merge_all(customers, orders, order_items, products, categories, reviews)

# ------------------------
# 4. Drop all missing values (strict version)
# ------------------------
merged_df = merged_df.dropna()
print(f"After dropping all missing values, remaining rows: {merged_df.shape[0]}")

# ------------------------
# 5. Clean outliers + visualize
# ------------------------
numeric_cols = ["price", "freight_value", "product_weight_g", 
                "product_length_cm", "product_height_cm", "product_width_cm",
                "Recency", "Frequency", "Monetary"]

merged_df = clean_outliers_with_boxplots(merged_df, numeric_cols)

# ------------------------
# 6. Feature engineering
# ------------------------
merged_df = feature_engineering(merged_df)

# ------------------------
# 7. Add RFM features
# ------------------------
merged_df = add_rfm_features(merged_df)

# ------------------------
# 8. Final encoding + scaling
# ------------------------
merged_df, encoders, scaler = final_preprocessing(merged_df)

# ------------------------
# 9. Verify final dataset
# ------------------------
verify_dataset(merged_df)

# ------------------------
# 10. Save Processed Data
# ------------------------
output_folder = "processed_data"
output_file = "preprocessed_olist_data_strict.csv"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

merged_df.to_csv(os.path.join(output_folder, output_file), index=False)
print(f"✅ Strict preprocessed dataset saved successfully at: {output_folder}/{output_file}")
