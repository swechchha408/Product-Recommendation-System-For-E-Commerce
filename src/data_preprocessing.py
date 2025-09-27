import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. Preprocess Customers
# =========================
def preprocess_customers(customers):
    customers = customers.drop_duplicates()
    customers = customers.fillna("Unknown")
    
    # Convert types
    customers["customer_id"] = customers["customer_id"].astype(str)
    customers["customer_unique_id"] = customers["customer_unique_id"].astype(str)
    customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str)
    customers["customer_city"] = customers["customer_city"].astype(str).str.lower().str.replace(" ", "_")
    customers["customer_state"] = customers["customer_state"].astype(str).str.lower().str.replace(" ", "_")
    
    print("🔍 Customers duplicates:", customers.duplicated().sum())
    print("🔍 Customers missing values:\n", customers.isnull().sum())
    
    return customers

# =========================
# 2. Preprocess Orders
# =========================
def preprocess_orders(orders):
    orders = orders.drop_duplicates()
    orders = orders.fillna(pd.NaT)
    
    # Convert types
    orders["order_id"] = orders["order_id"].astype(str)
    orders["customer_id"] = orders["customer_id"].astype(str)
    orders["order_status"] = orders["order_status"].astype(str)
    
    date_cols = ["order_purchase_timestamp", "order_approved_at", 
                 "order_delivered_carrier_date", "order_delivered_customer_date", 
                 "order_estimated_delivery_date"]
    
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors='coerce', dayfirst=True)
    
    orders = orders.dropna(subset=["order_purchase_timestamp"])
    
    print("🔍 Orders duplicates:", orders.duplicated().sum())
    print("🔍 Orders missing values:\n", orders.isnull().sum())
    
    return orders

# =========================
# 3. Preprocess Order Items
# =========================
def preprocess_order_items(order_items):
    order_items = order_items.drop_duplicates()
    order_items = order_items.fillna(0)
    
    order_items["order_id"] = order_items["order_id"].astype(str)
    order_items["product_id"] = order_items["product_id"].astype(str)
    order_items["seller_id"] = order_items["seller_id"].astype(str)
    order_items["order_item_id"] = order_items["order_item_id"].astype(int)
    order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce").fillna(0)
    order_items["freight_value"] = pd.to_numeric(order_items["freight_value"], errors="coerce").fillna(0)
    order_items["shipping_limit_date"] = pd.to_datetime(order_items["shipping_limit_date"], errors="coerce", dayfirst=True)
    
    print("🔍 Order Items duplicates:", order_items.duplicated().sum())
    print("🔍 Order Items missing values:\n", order_items.isnull().sum())
    
    return order_items

# =========================
# 4. Preprocess Products
# =========================
def preprocess_products(products):
    products = products.drop_duplicates()
    
    products["product_id"] = products["product_id"].astype(str)
    products["product_category_name"] = products["product_category_name"].fillna("unknown").astype(str).str.lower().str.replace(" ", "_")
    
    numeric_cols = ["product_name_lenght", "product_description_lenght", "product_photos_qty",
                    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    
    for col in numeric_cols:
        products[col] = pd.to_numeric(products[col], errors="coerce").fillna(products[col].median()).astype("Int64")
    
    print("🔍 Products duplicates:", products.duplicated().sum())
    print("🔍 Products missing values:\n", products.isnull().sum())
    
    return products

# =========================
# 5. Preprocess Categories
# =========================
def preprocess_categories(categories):
    categories = categories.drop_duplicates()
    categories["product_category_name"] = categories["product_category_name"].fillna("unknown").astype(str).str.lower().str.replace(" ", "_")
    categories["product_category_name_english"] = categories["product_category_name_english"].fillna("unknown").astype(str).str.lower().str.replace(" ", "_")
    
    print("🔍 Categories duplicates:", categories.duplicated().sum())
    print("🔍 Categories missing values:\n", categories.isnull().sum())
    
    return categories

# =========================
# 6. Preprocess Reviews
# =========================
def preprocess_reviews(reviews):
    reviews = reviews.drop_duplicates()
    
    reviews["review_id"] = reviews["review_id"].astype(str)
    reviews["order_id"] = reviews["order_id"].astype(str)
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce").fillna(0).astype("Int64")
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("No Review").astype(str)
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("No Review").astype(str)
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")
    
    print("🔍 Reviews duplicates:", reviews.duplicated().sum())
    print("🔍 Reviews missing values:\n", reviews.isnull().sum())
    
    return reviews

# =========================
# 7. Merge All Datasets
# =========================
def merge_all(customers, orders, order_items, products, categories, reviews):
    merged = (orders
              .merge(customers, on="customer_id", how="left")
              .merge(order_items, on="order_id", how="left")
              .merge(products, on="product_id", how="left")
              .merge(categories, left_on="product_category_name", right_on="product_category_name", how="left")
              .merge(reviews, on="order_id", how="left"))
    return merged

# =========================
# 8. Outlier removal + boxplots
# =========================
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def clean_outliers_with_boxplots(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = ["price", "freight_value", "product_weight_g", 
                        "product_length_cm", "product_height_cm", "product_width_cm",
                        "Recency", "Frequency", "Monetary"]

    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(8,4))
            sns.boxplot(x=df[col])
            plt.title(f'Before outlier removal: {col}')
            plt.show()
            
            df = remove_outliers_iqr(df, col)
            
            plt.figure(figsize=(8,4))
            sns.boxplot(x=df[col])
            plt.title(f'After outlier removal: {col}')
            plt.show()
    return df

# =========================
# 9. Feature Engineering
# =========================
def feature_engineering(df):
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    user_orders = df.groupby("customer_id")["order_id"].nunique().reset_index()
    user_orders.rename(columns={"order_id": "total_orders"}, inplace=True)
    df = df.merge(user_orders, on="customer_id", how="left")

    product_orders = df.groupby("product_id")["order_id"].nunique().reset_index()
    product_orders.rename(columns={"order_id": "total_product_orders"}, inplace=True)
    df = df.merge(product_orders, on="product_id", how="left")

    df["freight_to_price_ratio"] = df["freight_value"] / (df["price"] + 1e-5)
    review_agg = df.groupby("product_id")["review_score"].agg(["mean", "count"]).reset_index()
    review_agg.rename(columns={"mean": "avg_review_score", "count": "num_reviews"}, inplace=True)
    df = df.merge(review_agg, on="product_id", how="left")

    return df

# =========================
# 10. RFM Features
# =========================
def add_rfm_features(df):
    ref_date = df['order_purchase_timestamp'].max()
    rfm = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (ref_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()
    rfm.rename(columns={'order_purchase_timestamp': 'Recency',
                        'order_id': 'Frequency',
                        'price': 'Monetary'}, inplace=True)
    df = df.merge(rfm, on='customer_id', how='left')
    return df

# =========================
# 11. Final Preprocessing: Encoding + Scaling
# =========================
def final_preprocessing(df):
    cat_cols = ["customer_id", "product_id", "product_category_name_english"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    num_cols = ["price", "freight_value", "total_orders", "total_product_orders",
                "Recency", "Frequency", "Monetary", "freight_to_price_ratio",
                "avg_review_score", "num_reviews"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))

    return df, encoders, scaler

# =========================
# 12. Verification
# =========================
def verify_dataset(df):
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())
    print("Data types:\n", df.dtypes)
