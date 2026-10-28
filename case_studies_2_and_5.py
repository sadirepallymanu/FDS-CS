
# Fundamentals of Data Science: Case Studies 2 and 5


# 1. Setup

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

os.makedirs('figures', exist_ok=True)
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': 'black', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'savefig.dpi': 200,
    'axes.prop_cycle': cycler(color=['black', '0.35', '0.55', '0.75']),
})
SHADES = ['black', '0.35', '0.6', '0.8']
MARKERS = ['o', 's', '^', 'D']

def corr_heatmap(df, title, path, size=(8, 6.5)):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(corr.abs().values, cmap='Greys', vmin=0, vmax=1.4)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > 0.7 else 'black')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.show()

# Case Study 2: Clustering Analysis


# 2.1 Loading the Data

airline = pd.read_excel('data/EastWestAirlines.xlsx', sheet_name='data')
print(airline.shape)
print(airline.dtypes)
print(airline.describe().T.round(2).to_string())

# 2.2 Data Preprocessing

features = [c for c in airline.columns if c not in ('ID#', 'Award?') and not c.startswith('cc')]
print('Missing values:', int(airline.isnull().sum().sum()))
print('Duplicate customers:', int(airline.drop(columns='ID#').duplicated().sum()))
airline = airline.drop_duplicates(subset=[c for c in airline.columns if c != 'ID#']).reset_index(drop=True)

z = (airline[features] - airline[features].mean()) / airline[features].std()
clean = airline[(z.abs() < 5).all(axis=1)].reset_index(drop=True)
print('Rows before outlier removal:', len(airline))
print('Rows after outlier removal:', len(clean))
print('Rows removed:', len(airline) - len(clean))

X = StandardScaler().fit_transform(clean[features])
print('Scaled matrix shape:', X.shape)
print(pd.DataFrame(X, columns=features).describe().loc[['mean', 'std']].round(2).to_string())

# 2.3 Exploratory Data Analysis

cols = ['Balance', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll']
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, col in zip(axes.ravel(), cols):
    ax.hist(clean[col], bins=30, color='0.5', edgecolor='black')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Number of customers')
fig.tight_layout()
fig.savefig('figures/cs2_histograms.png')
plt.show()
print(clean[cols].skew().round(2).to_string())
corr_heatmap(clean[features], 'Correlation Between Customer Features', 'figures/cs2_heatmap.png', size=(8.5, 7))
print(clean[features].corr().round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(clean['Balance'], clean['Bonus_miles'], s=8, color='0.3')
axes[0].set_title('Balance vs Bonus Miles')
axes[0].set_xlabel('Balance')
axes[0].set_ylabel('Bonus miles')
axes[1].scatter(clean['Flight_trans_12'], clean['Flight_miles_12mo'], s=8, color='0.3')
axes[1].set_title('Flight Transactions vs Flight Miles')
axes[1].set_xlabel('Flight transactions in 12 months')
axes[1].set_ylabel('Flight miles in 12 months')
fig.tight_layout()
fig.savefig('figures/cs2_scatter_eda.png')
plt.show()

# 2.4 K-Means Clustering

ks = list(range(2, 11))
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
print(pd.DataFrame({'K': ks, 'Inertia': np.round(inertia, 0), 'Silhouette': np.round(sil, 3)}).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(ks, inertia, marker='o', color='black')
axes[0].set_title('Elbow Curve for K-Means')
axes[0].set_xlabel('Number of clusters (K)')
axes[0].set_ylabel('Inertia')
axes[1].plot(ks, sil, marker='s', color='black')
axes[1].set_title('Silhouette Score for Different K')
axes[1].set_xlabel('Number of clusters (K)')
axes[1].set_ylabel('Silhouette score')
fig.tight_layout()
fig.savefig('figures/cs2_kmeans_elbow.png')
plt.show()
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
clean['KMeans'] = kmeans.labels_
print(clean['KMeans'].value_counts().sort_index().to_string())

# 2.5 Hierarchical Clustering

Z = linkage(X, method='ward')
fig, ax = plt.subplots(figsize=(10, 4.5))
dendrogram(Z, truncate_mode='lastp', p=30, color_threshold=0, above_threshold_color='black', ax=ax)
ax.set_title('Dendrogram Using Ward Linkage (Last 30 Merged Clusters)')
ax.set_xlabel('Cluster size or customer index')
ax.set_ylabel('Merge distance')
fig.tight_layout()
fig.savefig('figures/cs2_dendrogram.png')
plt.show()
rows = []
for method in ['ward', 'complete', 'average', 'single']:
    labels = AgglomerativeClustering(n_clusters=4, linkage=method).fit_predict(X)
    rows.append({'Linkage': method,
                 'Silhouette': round(silhouette_score(X, labels), 3),
                 'Davies-Bouldin': round(davies_bouldin_score(X, labels), 3),
                 'Cluster sizes': sorted(np.bincount(labels).tolist(), reverse=True)})
print(pd.DataFrame(rows).to_string(index=False))

hier = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
clean['Hierarchical'] = hier.labels_
print(clean['Hierarchical'].value_counts().sort_index().to_string())

# 2.6 DBSCAN Clustering

nn = NearestNeighbors(n_neighbors=10).fit(X)
distances, _ = nn.kneighbors(X)
kdist = np.sort(distances[:, -1])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(kdist, color='black')
ax.set_title('K-Distance Plot (10th Nearest Neighbour)')
ax.set_xlabel('Customers sorted by distance')
ax.set_ylabel('Distance to 10th nearest neighbour')
fig.tight_layout()
fig.savefig('figures/cs2_kdistance.png')
plt.show()
rows = []
for eps in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for ms in [5, 10, 20]:
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        score = silhouette_score(X[mask], labels[mask]) if n_clusters > 1 else np.nan
        rows.append({'eps': eps, 'min_samples': ms, 'clusters': n_clusters,
                     'noise': int((~mask).sum()), 'silhouette': round(score, 3) if n_clusters > 1 else np.nan})
grid = pd.DataFrame(rows)
print(grid.to_string(index=False))

valid = grid[(grid['clusters'] >= 2) & (grid['noise'] <= 0.1 * len(X))]
best = valid.sort_values('silhouette', ascending=False).iloc[0]
print(best.to_string())

dbscan = DBSCAN(eps=float(best['eps']), min_samples=int(best['min_samples'])).fit(X)
clean['DBSCAN'] = dbscan.labels_
print(clean['DBSCAN'].value_counts().sort_index().to_string())

# 2.7 Cluster Visualization

pca = PCA(n_components=2, random_state=42)
P = pca.fit_transform(X)
print(pca.explained_variance_ratio_.round(3), pca.explained_variance_ratio_.sum().round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
for ax, col, name in zip(axes, ['KMeans', 'Hierarchical', 'DBSCAN'], ['K-Means', 'Hierarchical (Ward)', 'DBSCAN']):
    labels = clean[col].values
    for i, lab in enumerate(sorted(set(labels))):
        m = labels == lab
        if lab == -1:
            ax.scatter(P[m, 0], P[m, 1], s=14, marker='x', color='black', label='Noise')
        else:
            ax.scatter(P[m, 0], P[m, 1], s=9, marker=MARKERS[lab % 4], color=SHADES[lab % 4],
                       edgecolor='black', linewidth=0.2, label=f'Cluster {lab}')
    ax.set_title(f'{name} Clusters (PCA View)')
    ax.set_xlabel('Principal component 1')
    ax.legend(markerscale=1.6, fontsize=8)
axes[0].set_ylabel('Principal component 2')
fig.tight_layout()
fig.savefig('figures/cs2_cluster_scatter.png')
plt.show()

# 2.8 Cluster Profiles

profile_cols = ['Balance', 'Qual_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award?']
for col in ['KMeans', 'Hierarchical', 'DBSCAN']:
    prof = clean.groupby(col)[profile_cols].mean().round(2)
    prof.insert(0, 'Customers', clean.groupby(col).size())
    print(col)
    print(prof.to_string())
    print()
medians = clean.groupby('KMeans')[['Balance', 'Bonus_miles', 'Flight_miles_12mo', 'Days_since_enroll']].median()
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, medians.columns):
    ax.bar(medians.index.astype(str), medians[col], color='0.5', edgecolor='black')
    ax.set_title(f'Median {col}')
    ax.set_xlabel('K-Means cluster')
    ax.set_ylabel(col)
fig.tight_layout()
fig.savefig('figures/cs2_cluster_profiles.png')
plt.show()

# 2.9 Evaluation Summary

summary = []
for col, name in [('KMeans', 'K-Means'), ('Hierarchical', 'Hierarchical (Ward)'), ('DBSCAN', 'DBSCAN')]:
    labels = clean[col].values
    m = labels != -1
    summary.append({'Method': name, 'Clusters': len(set(labels[m])), 'Noise points': int((~m).sum()),
                    'Silhouette': round(silhouette_score(X[m], labels[m]), 3),
                    'Davies-Bouldin': round(davies_bouldin_score(X[m], labels[m]), 3)})
print(pd.DataFrame(summary).to_string(index=False))

# Case Study 5: Wine Quality Assessment


# 3.1 Loading the Data

red = pd.read_csv('data/winequality-red.csv', sep=';')
white = pd.read_csv('data/winequality-white.csv', sep=';')
wine = pd.concat([red, white], ignore_index=True)
wine_type = np.array(['red'] * len(red) + ['white'] * len(white))
print(red.shape, white.shape, wine.shape)
print('Missing values:', int(wine.isnull().sum().sum()))
print(wine.describe().T.round(3).to_string())

# 3.2 Task 1: Statistical Analysis

attrs = ['alcohol', 'volatile acidity', 'residual sugar']
rows = []
for a in attrs:
    rows.append({'Attribute': a, 'Mean': wine[a].mean(), 'Median': wine[a].median(),
                 'Mode': wine[a].mode().iloc[0], 'Std Dev': wine[a].std(),
                 'Skewness': wine[a].skew(), 'CV (%)': wine[a].std() / wine[a].mean() * 100})
desc = pd.DataFrame(rows).set_index('Attribute').round(3)
print(desc.to_string())
print('Highest relative variability:', desc['CV (%)'].idxmax())

r, p = stats.pearsonr(wine['alcohol'], wine['quality'])
print('Pearson r (alcohol, quality):', round(r, 4))
print('p-value:', p)

# 3.3 Task 2: Exploratory Data Analysis

counts = wine['quality'].value_counts().sort_index()
print(counts.to_string())
print((counts / len(wine) * 100).round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(counts.index.astype(str), counts.values, color='0.5', edgecolor='black')
for x, v in zip(range(len(counts)), counts.values):
    ax.text(x, v + 20, str(v), ha='center')
ax.set_title('Histogram of Wine Quality Scores')
ax.set_xlabel('Quality score')
ax.set_ylabel('Number of wines')
fig.tight_layout()
fig.savefig('figures/cs5_quality_histogram.png')
plt.show()
levels = sorted(wine['quality'].unique())
data = [wine.loc[wine['quality'] == q, 'alcohol'] for q in levels]
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(data, tick_labels=[str(q) for q in levels], patch_artist=True,
           boxprops=dict(facecolor='0.8', edgecolor='black'), medianprops=dict(color='black', linewidth=2),
           whiskerprops=dict(color='black'), capprops=dict(color='black'),
           flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=4))
ax.set_title('Alcohol Content by Wine Quality Score')
ax.set_xlabel('Quality score')
ax.set_ylabel('Alcohol (% volume)')
fig.tight_layout()
fig.savefig('figures/cs5_alcohol_boxplot.png')
plt.show()

rows = []
for q, d in zip(levels, data):
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    out = int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum())
    rows.append({'Quality': q, 'Wines': len(d), 'Median alcohol': round(d.median(), 2), 'Outliers': out})
print(pd.DataFrame(rows).to_string(index=False))
corr_heatmap(wine, 'Correlation Heatmap of Wine Attributes and Quality', 'figures/cs5_heatmap.png', size=(9.5, 8))
qcorr = wine.corr()['quality'].drop('quality')
print(qcorr.round(3).sort_values(key=abs, ascending=False).to_string())
print('Top two:', list(qcorr.abs().sort_values(ascending=False).index[:2]))

# 3.4 Task 3: Predictive Modelling

inputs = [c for c in wine.columns if c != 'quality']
X_wine = wine[inputs]
y_wine = wine['quality']
X_train, X_test, y_train, y_test, type_train, type_test = train_test_split(
    X_wine, y_wine, wine_type, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
print('MAE :', round(mae, 4))
print('MSE :', round(mse, 4))
print('RMSE:', round(rmse, 4))
print('R2  :', round(r2, 4))
print('Baseline RMSE (always predict the training mean):',
      round(np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))), 4))

rounded = np.clip(np.round(pred), 3, 9)
print('Exact match after rounding (%):', round((rounded == y_test).mean() * 100, 1))
print('Within one point (%):', round((np.abs(rounded - y_test) <= 1).mean() * 100, 1))
scaler = StandardScaler().fit(X_train)
std_model = LinearRegression().fit(scaler.transform(X_train), y_train)
std_coef = pd.Series(std_model.coef_, index=inputs)
print(std_coef.sort_values(key=abs, ascending=False).round(3).to_string())

vif = {}
for col in inputs:
    others = [c for c in inputs if c != col]
    r2_col = LinearRegression().fit(X_train[others], X_train[col]).score(X_train[others], X_train[col])
    vif[col] = 1 / (1 - r2_col)
print(pd.Series(vif).sort_values(ascending=False).round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
order = std_coef.sort_values(key=abs)
axes[0].barh(order.index, order.values, color='0.5', edgecolor='black')
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title('Standardized Regression Coefficients')
axes[0].set_xlabel('Change in quality per one standard deviation')
rng = np.random.default_rng(42)
axes[1].scatter(y_test + rng.uniform(-0.15, 0.15, len(y_test)), pred, s=10, color='0.3', alpha=0.6)
axes[1].plot([3, 9], [3, 9], color='black', linestyle='--', label='Perfect prediction')
axes[1].set_title('Predicted vs Actual Quality (Test Set)')
axes[1].set_xlabel('Actual quality score')
axes[1].set_ylabel('Predicted quality score')
axes[1].legend()
fig.tight_layout()
fig.savefig('figures/cs5_model_plots.png')
plt.show()
for t in ['red', 'white']:
    idx = wine_type == t
    Xt, Xs, yt, ys = train_test_split(X_wine[idx], y_wine[idx], test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xt, yt)
    p = m.predict(Xs)
    print(t, 'MAE', round(mean_absolute_error(ys, p), 4), 'RMSE', round(np.sqrt(mean_squared_error(ys, p)), 4), 'R2', round(r2_score(ys, p), 4))

# 3.5 Task 4: Business Decision-Making

summary = pd.DataFrame({'Correlation with quality': qcorr.round(3), 'Standardized coefficient': std_coef.round(3)})
print(summary.sort_values('Standardized coefficient', key=abs, ascending=False).to_string())
print('Test R2:', round(r2, 4), '| Test RMSE:', round(rmse, 4), '| Test MAE:', round(mae, 4))
