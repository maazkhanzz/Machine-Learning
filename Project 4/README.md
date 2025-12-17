# Project 4 — Apartment Rental Clustering 

## Overview
This project explores **unsupervised clustering** on rental apartment listings to discover natural groups such as budget rentals, mid-range units, and luxury properties.
I compare three clustering approaches:
- **K-Means**
- **Hierarchical Agglomerative Clustering (Ward linkage)**
- **DBSCAN** (to detect outliers/noise)

## Dataset
- **Source:** UCI Machine Learning Repository — *Apartment for Rent Classified Dataset*
- **Subset used:** `apt_10k.csv` (10,000 listings)
- The raw dataset is **semicolon-separated** and uses `"null"` for missing values. I convert `"null"` to `NaN` and remove incomplete rows before clustering.

### Features used
**Numeric**
- `price`, `square_feet`, `bathrooms`, `bedrooms`, `latitude`, `longitude`

**Categorical**
- `category`, `currency`, `fee`, `has_photo`, `pets_allowed`, `price_type`, `cityname`, `state`, `source`

## Methods
### Preprocessing
- Numeric features are standardized using **StandardScaler**
- Categorical features are one-hot encoded using **OneHotEncoder(handle_unknown="ignore")**
- A **ColumnTransformer** applies both transformations consistently

### Models compared
- **K-Means:** tested `K = 2..10` using SSE (inertia) + silhouette score
- **Hierarchical:** Ward linkage with `n_clusters = 5`
- **DBSCAN:** tuned with `eps = 1.3`, `min_samples = 20`

## Results (add figures below)
> I will add plots here (Elbow, Silhouette, PCA cluster visualizations, Dendrogram, DBSCAN noise plot).
