import numpy as np
import pandas as pd
import joblib as jlb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import Counter
import sys
import os

FEATURES = ['Income', 'Age', 'Total_Spend']

# Construct paths
DATASET_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'datasets', 'marketing_campaign.csv')
OUTPUT_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Output', '')

# Seed for reproducibility
RANDOM_SEED = 42

def load_data():
    """Load and normalize customer segmentation data"""
    # Read raw data
    data = pd.read_csv(DATASET_PATH)
    
    # Data preprocessing
    data['Income'] = data['Income'].fillna(data['Income'].median())
    
    data["Living_With"] = data["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner", 
        "Absurd": "Alone", 
        "Widow": "Alone", 
        "YOLO": "Alone", 
        "Divorced": "Alone", 
        "Single": "Alone",
    })
    
    data['Children'] = data['Kidhome'] + data['Teenhome']
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    
    data["Education"] = data["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", 
        "PhD": "Postgraduate"
    })
    
    data['Age'] = datetime.now().year - data['Year_Birth']
    data['Total_Spend'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + 
                          data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])
    
    # Select features for clustering
    selected_data = data[FEATURES].copy()
    print(f"✓ Data loaded: {selected_data.shape[0]} samples, {selected_data.shape[1]} features")
    
    # Normalize features for fair comparison
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(selected_data.values)
    print(f"✓ Data normalized using StandardScaler")
    
    return X_normalized


def load_runtime_metadata():
    """Load runtime data from training metadata JSON file"""
    try:
        metadata_path = f'{OUTPUT_DIR}training_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Training metadata loaded from {metadata_path}")
        return metadata.get('runtimes', {})
    except FileNotFoundError:
        print(f"⚠ Training metadata not found. Run retrain_models.py first.")
        return {}


def load_models(X):
    """Load pre-trained models from Output directory, handling missing algorithms gracefully"""
    print("\nLoading pre-trained models...")
    models = {}
    
    try:
        clarans = jlb.load(f'{OUTPUT_DIR}clarans.mdl')
        # Handle both method calls and direct attribute access
        labels = clarans.get_labels() if hasattr(clarans, 'get_labels') else getattr(clarans, 'labels', None)
        centers = clarans.get_medoids() if hasattr(clarans, 'get_medoids') else getattr(clarans, 'medoids', None)
        
        if labels is not None and centers is not None:
            models['CLARANS'] = {
                'labels': labels,
                'centers': centers,
                'medoid_based': True
            }
            print("  ✓ CLARANS loaded")
        else:
            print("  ✗ CLARANS: Missing labels or centers")
    except Exception as e:
        print(f"  ✗ CLARANS failed: {e}")
    
    try:
        kmeans = jlb.load(f'{OUTPUT_DIR}kMeans.mdl')
        labels = kmeans.get_labels() if hasattr(kmeans, 'get_labels') else getattr(kmeans, 'labels', None)
        centers = kmeans.get_centroids() if hasattr(kmeans, 'get_centroids') else getattr(kmeans, 'centroids', None)
        
        if labels is not None and centers is not None:
            models['K-Means'] = {
                'labels': labels,
                'centers': centers,
                'medoid_based': False
            }
            print("  ✓ K-Means loaded")
        else:
            print("  ✗ K-Means: Missing labels or centers")
    except Exception as e:
        print(f"  ✗ K-Means failed: {e}")
    
    try:
        pam = jlb.load(f'{OUTPUT_DIR}PAM.mdl')
        labels = pam.get_labels() if hasattr(pam, 'get_labels') else getattr(pam, 'labels_', None)
        centers = pam.get_medoids() if hasattr(pam, 'get_medoids') else getattr(pam, 'medoids_', None)
        
        if labels is not None and centers is not None:
            models['PAM'] = {
                'labels': labels,
                'centers': centers,
                'medoid_based': True
            }
            print("  ✓ PAM loaded")
        else:
            print("  ✗ PAM: Missing labels or centers")
    except Exception as e:
        print(f"  ✗ PAM failed: {e}")
    
    # DBSCAN - train directly on normalized data for consistency
    try:
        dbscan = DBSCAN(eps=0.25, min_samples=10)
        labels = dbscan.fit_predict(X)
        models['DBSCAN'] = {
            'labels': labels,
            'centers': None,
            'medoid_based': False
        }
        print("  ✓ DBSCAN trained on normalized data")
    except Exception as e:
        print(f"  ✗ DBSCAN failed: {e}")
    
    return models

def compute_silhouette(X, labels):
    """Silhouette Score: -1 to 1, higher is better"""
    if len(np.unique(labels)) < 2:
        return np.nan
    
    valid_mask = labels != -1
    if np.sum(valid_mask) < 2:
        return np.nan
    
    try:
        return silhouette_score(X[valid_mask], labels[valid_mask])
    except:
        return np.nan


def compute_davies_bouldin(X, labels):
    """Davies-Bouldin Index: 0 to inf, lower is better"""
    if len(np.unique(labels)) < 2:
        return np.nan
    
    valid_mask = labels != -1
    if np.sum(valid_mask) < 2:
        return np.nan
    
    try:
        return davies_bouldin_score(X[valid_mask], labels[valid_mask])
    except:
        return np.nan


def compute_calinski_harabasz(X, labels):
    """Calinski-Harabasz Score: higher is better, ratio of between-cluster to within-cluster variance"""
    if len(np.unique(labels)) < 2:
        return np.nan
    
    valid_mask = labels != -1
    if np.sum(valid_mask) < 2:
        return np.nan
    
    try:
        return calinski_harabasz_score(X[valid_mask], labels[valid_mask])
    except:
        return np.nan


def compute_cost(X, labels, centers, is_medoid_based=False):
    """
    Total cost: sum of distances from each point to nearest center.
    
    For medoid-based algorithms (CLARANS, PAM): centers are actual data points (medoids)
    For centroid-based algorithms (K-Means): centers might be synthetic points
    
    Cost computation is the same for both, but interpretation differs.
    """
    if centers is None:
        return np.nan
    
    try:
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        return np.sum(np.min(distances, axis=1))
    except:
        return np.nan


def compare_algorithms(X, models, runtimes):
    """Compute all metrics for comparison"""
    print("\nComputing metrics...")
    
    results = []
    for algo_name, model_info in models.items():
        labels = model_info['labels']
        centers = model_info['centers']
        is_medoid_based = model_info['medoid_based']
        
        silhouette = compute_silhouette(X, labels)
        davies_bouldin = compute_davies_bouldin(X, labels)
        calinski_harabasz = compute_calinski_harabasz(X, labels)
        cost = compute_cost(X, labels, centers, is_medoid_based)
        runtime = runtimes.get(algo_name, np.nan)
        n_clusters = len(np.unique(labels[labels != -1]))
        
        results.append({
            'Algorithm': algo_name,
            'Silhouette': silhouette,
            'Davies-Bouldin': davies_bouldin,
            'Calinski-Harabasz': calinski_harabasz,
            'Cost': cost,
            'Time (s)': runtime,
            'Clusters': n_clusters
        })
        
        print(f"  {algo_name}: Silhouette={silhouette:.4f}, CH={calinski_harabasz:.1f}, Cost={cost:.0f}")
    
    df = pd.DataFrame(results)
    
    # Sort by Silhouette score descending (higher is better)
    df = df.sort_values('Silhouette', ascending=False, na_position='last').reset_index(drop=True)
    
    return df

def print_results(df):
    """Print comparison table with neutral, academic analysis"""
    print("\n" + "=" * 100)
    print("CLUSTERING ALGORITHMS COMPARISON RESULTS")
    print("=" * 100)
    
    print("\nMETRICS TABLE (sorted by Silhouette Score, descending):")
    print("-" * 100)
    display_df = df[['Algorithm', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Cost', 'Time (s)', 'Clusters']].copy()
    display_df['Silhouette'] = display_df['Silhouette'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    display_df['Davies-Bouldin'] = display_df['Davies-Bouldin'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    display_df['Calinski-Harabasz'] = display_df['Calinski-Harabasz'].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A")
    display_df['Cost'] = display_df['Cost'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    display_df['Time (s)'] = display_df['Time (s)'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    print(display_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("METRIC DEFINITIONS")
    print("=" * 100)
    print("""
  • Silhouette Score: Range [-1, 1]. Higher is better. 
    Measures how similar objects are to their own cluster vs other clusters.
    
  • Davies-Bouldin Index: Range [0, ∞). Lower is better.
    Ratio of within-cluster to between-cluster distances. Smaller = better separation.
    
  • Calinski-Harabasz Score: Range [0, ∞). Higher is better.
    Ratio of between-cluster to within-cluster variance. Higher = denser, better-separated clusters.
    
  • Cost: Total distance from all points to their nearest center.
    Lower is better. Meaningful mainly for medoid-based algorithms (PAM, CLARANS).
    
  • Time: Actual execution time during training. Lower is better.
    
  • Clusters: Number of clusters identified by the algorithm.
""")
    
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    best_silhouette = df.loc[df['Silhouette'].idxmax()] if not df['Silhouette'].isna().all() else None
    best_davies = df.loc[df['Davies-Bouldin'].idxmin()] if not df['Davies-Bouldin'].isna().all() else None
    best_ch = df.loc[df['Calinski-Harabasz'].idxmax()] if not df['Calinski-Harabasz'].isna().all() else None
    
    if best_silhouette is not None:
        print(f"\nBest Silhouette Score: {best_silhouette['Algorithm']} ({best_silhouette['Silhouette']:.4f})")
    if best_davies is not None:
        print(f"Best Davies-Bouldin Index: {best_davies['Algorithm']} ({best_davies['Davies-Bouldin']:.4f})")
    if best_ch is not None:
        print(f"Best Calinski-Harabasz Score: {best_ch['Algorithm']} ({best_ch['Calinski-Harabasz']:.1f})")
    
    print("\n" + "=" * 100 + "\n")


def save_results(df):
    """Save results to CSV"""
    csv_file = f'{OUTPUT_DIR}algorithm_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Results saved: {csv_file}")
    return csv_file

def create_visualizations(df):
    """Create comparison charts"""
    print("\nCreating visualizations...")
    
    # Neutral color scheme (not highlighting CLARANS)
    colors = ['#3498db' for _ in df['Algorithm']]  # Uniform blue color
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clustering Algorithms Comparison - Key Metrics', fontsize=16, fontweight='bold')
    
    # 1. Silhouette
    axes[0, 0].bar(df['Algorithm'], df['Silhouette'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Silhouette Score', fontweight='bold')
    axes[0, 0].set_title('Silhouette Score (↑ Higher is Better)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Silhouette']):
        if not pd.isna(v):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Davies-Bouldin
    axes[0, 1].bar(df['Algorithm'], df['Davies-Bouldin'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Davies-Bouldin Index', fontweight='bold')
    axes[0, 1].set_title('Davies-Bouldin Index (↓ Lower is Better)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Davies-Bouldin']):
        if not pd.isna(v):
            axes[0, 1].text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Calinski-Harabasz
    axes[1, 0].bar(df['Algorithm'], df['Calinski-Harabasz'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Calinski-Harabasz Score', fontweight='bold')
    axes[1, 0].set_title('Calinski-Harabasz Score (↑ Higher is Better)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Calinski-Harabasz']):
        if not pd.isna(v):
            axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Execution Time
    axes[1, 1].bar(df['Algorithm'], df['Time (s)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Time (seconds)', fontweight='bold')
    axes[1, 1].set_title('Execution Time (↓ Lower is Better)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Time (s)']):
        if not pd.isna(v):
            axes[1, 1].text(i, v + 0.005, f'{v:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved: {OUTPUT_DIR}algorithm_comparison.png")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "=" * 100)
    print("CLUSTERING ALGORITHMS COMPARISON (with reproducibility controls)")
    print("=" * 100)
    print(f"\nRandom Seed (for reproducibility): {RANDOM_SEED}\n")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load data
    X = load_data()
    
    # Load runtime metadata
    runtimes = load_runtime_metadata()
    
    # Load models (pass X for DBSCAN training on normalized data)
    models = load_models(X)
    
    if not models:
        print("\n✗ No models found! Run retrain_models.py first.")
        print("   Command: python retrain_models.py")
        return
    
    print(f"\n✓ Loaded {len(models)} algorithms: {', '.join(models.keys())}")
    
    # Compare
    df = compare_algorithms(X, models, runtimes)
    
    # Display results
    print_results(df)
    
    # Save & visualize
    save_results(df)
    create_visualizations(df)
    
    print("✓ Comparison complete! Check Output/ folder for results.")


if __name__ == '__main__':
    main()
