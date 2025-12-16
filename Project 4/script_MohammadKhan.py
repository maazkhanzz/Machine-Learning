import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# I had to download the file from UCI, it's semicolon separated and has weird quotes
df = pd.read_csv("apt_10k.csv", sep=";", encoding="latin1") 

# replace "null" strings with real NaN
df.replace("null", np.nan, inplace=True)

print("original shape:", df.shape)
print("columns:", list(df.columns)) # just to see what we have

# PART 1: Choose features and clean data
# i picked these because price, size, location obviously matter
# city and state probably group similar areas


numeric_cols = ["price", "square_feet", "bathrooms", "bedrooms", "latitude", "longitude"]
categorical_cols = ["category", "currency", "fee", "has_photo", "pets_allowed", "price_type", "cityname", "state", "source"]

# keep only these columns and drop rows with missing values 
df = df[numeric_cols + categorical_cols].dropna() 
print("after dropping NaNs:", df.shape)



# PART 2: Preprocessing - scale numbers and one-hot encode categories

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
]) 

X = preprocessor.fit_transform(df)
print("final feature matrix shape:", X.shape) # should be around (9000+, lots) because of one-hot



# PART 3: K-Means + elbow + silhouette to pick K

print("\n--- Running K-Means ---")
sse = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1) 
    labels = kmeans.fit_predict(X)
    sse.append(kmeans.inertia_)
    sil = silhouette_score(X, labels)
    silhouettes.append(sil)
    print(f"k={k} -> SSE={kmeans.inertia_:.0f}, silhouette={sil:.4f}") 

# elbow plot
plt.figure()
plt.plot(K_range, sse, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("SSE")
plt.savefig("elbow.png")
plt.show() 

# silhouette plot
plt.figure()
plt.plot(K_range, silhouettes, marker='o', color='orange')
plt.title("Silhouette Scores")
plt.xlabel("K")
plt.ylabel("Score")
plt.savefig("silhouette.png")
plt.show()

best_k = 5 # from the plots, k=5 or 6 looked best
print("I'm going with k =", best_k)

# final K-Means with best k
kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=999)
kmeans_labels = kmeans_final.fit_predict(X) 


# PART 4: PCA visualization for K-Means

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8,6))
for i in range(best_k):
    plt.scatter(X_2d[kmeans_labels==i, 0], X_2d[kmeans_labels==i, 1], label=f"Cluster {i}", s=15)
plt.title(f"K-Means with k={best_k} (PCA 2D)")
plt.legend()
plt.savefig("kmeans_pca.png")
plt.show() 



# PART 5: Hierarchical clustering (Ward)

print("\n--- Hierarchical (Ward) ---")
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_labels = hier.fit_predict(X) 

# PCA plot again
plt.figure(figsize=(8,6))
for i in range(best_k):
    plt.scatter(X_2d[hier_labels==i, 0], X_2d[hier_labels==i, 1], label=f"Cluster {i}", s=15)
plt.title("Hierarchical Ward (PCA)")
plt.legend()
plt.savefig("hierarchical_pca.png")
plt.show() 

# PART 6: DBSCAN (this one was tricky)

print("\n--- DBSCAN ---")
# tried eps=1.0 first -> too much noise, 1.3 seems better
dbscan = DBSCAN(eps=1.3, min_samples=20) # tweaked these values
db_labels = dbscan.fit_predict(X) 

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print("DBSCAN found", n_clusters, "clusters and", n_noise, "noise points")

# PCA plot with noise in black
plt.figure(figsize=(8,6))
unique = np.unique(db_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
for lab, col in zip(unique, colors):
    if lab == -1:
        plt.scatter(X_2d[db_labels==-1, 0], X_2d[db_labels==-1, 1], 
                    c='black', s=15, label="Noise")
    else:
        plt.scatter(X_2d[db_labels==lab, 0], X_2d[db_labels==lab, 1], 
                    c=[col], s=15, label=f"Cluster {lab}")
plt.title("DBSCAN (PCA)")
plt.legend()
plt.savefig("dbscan_pca.png")
plt.show() 

# PART 7: Cluster profiles - only for K-Means
df["kmeans_cluster"] = kmeans_labels
print("\nK-Means Cluster Profiles (average values):")
profile = df.groupby("kmeans_cluster")[numeric_cols].mean().round(2)
profile["count"] = df["kmeans_cluster"].value_counts().sort_index()
print(profile) 
profile.to_csv("kmeans_profiles.csv")

# dendrogram on small sample because full one looks bad
sample_X = X[np.random.choice(X.shape[0], 500, replace=False)]
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(sample_X, method="ward")
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Dendrogram (Ward) - truncated")
plt.savefig("dendrogram.png")
plt.show() 

print("\nall done! files saved:")
print("elbow.png, silhouette.png, kmeans_pca.png, hierarchical_pca.png, dbscan_pca.png, dendrogram.png, kmeans_profiles.csv")
