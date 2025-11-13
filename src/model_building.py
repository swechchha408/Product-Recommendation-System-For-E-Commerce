import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'interaction_table_encoded_train.csv')

# Load preprocessed data
data = pd.read_csv(data_path)

# Extract relevant columns
user_ids = data['customer_id_encoded'].values
product_ids = data['product_id_encoded'].values
ratings = data['interaction'].values  # can be 1/0 or rating value

# Get number of unique users/products
num_users = len(np.unique(user_ids))
num_products = len(np.unique(product_ids))

# Define embedding dimension
embedding_dim = 64

# User input and embedding
user_input = keras.Input(shape=(1,), name="user_input")
user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_dim, name="user_embedding")(user_input)
user_vec = layers.Flatten()(user_embedding)

# Product input and embedding
product_input = keras.Input(shape=(1,), name="product_input")
product_embedding = layers.Embedding(input_dim=num_products, output_dim=embedding_dim, name="product_embedding")(product_input)
product_vec = layers.Flatten()(product_embedding)

# Dot product to get interaction score
dot_product = layers.Dot(axes=1)([user_vec, product_vec])

# Optional dense layer (to improve learning)
x = layers.Dense(64, activation='relu')(dot_product)
output = layers.Dense(1, activation='sigmoid')(x)

# Define model
model = keras.Model(inputs=[user_input, product_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train-test split (you can load from your data_splits folder)
train_path = os.path.join(BASE_DIR, 'data_splits', 'train_interaction.csv')
test_path = os.path.join(BASE_DIR, 'data_splits', 'test_interaction.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Prepare training and testing inputs
x_train = [train_data['customer_id_encoded'].values, train_data['product_id_encoded'].values]
y_train = train_data['interaction'].values

x_test = [test_data['customer_id_encoded'].values, test_data['product_id_encoded'].values]
y_test = test_data['interaction'].values

# Train model
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                    epochs=10, batch_size=256, verbose=1)

# Save the model
model.save(os.path.join(BASE_DIR, 'model', 'recommendation_model.h5'))

print("✅ Model training completed and saved successfully!")
