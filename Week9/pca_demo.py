import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("iris.csv")

selected_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df_selected = df[selected_columns]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected.values)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the PCA output
principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"], index=df.index)

print("Original Data:")
print(df.head())
print("\nPrincipal Components:")
principal_df["species"] = df["species"]
print(principal_df.head())

# For future predictions
sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)
sample_after_pca = pca.transform(sample_scaled)

print(sample_after_pca)
