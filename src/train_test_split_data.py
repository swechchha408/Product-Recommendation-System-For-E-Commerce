import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Load encoded interaction table
interaction_table = pd.read_csv("data/interaction_table_encoded_train.csv")

# ✅ Create new folder for split data (if not exists)
output_folder = "data_splits"
os.makedirs(output_folder, exist_ok=True)

print(f"📁 Output folder ready: {output_folder}")

# ✅ Train-Test Split (80% train, 20% test)
train_data, test_data = train_test_split(
    interaction_table,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ✅ Save split data into the new folder
train_path = os.path.join(output_folder, "train_interaction.csv")
test_path = os.path.join(output_folder, "test_interaction.csv")

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print("\n✅ Train/Test Split Completed!")
print(f"🔹 Train size: {len(train_data)} rows")
print(f"🔹 Test size : {len(test_data)} rows")
print(f"✅ Files saved:\n  - {train_path}\n  - {test_path}")
