import numpy as np
import pandas as pd
import joblib as jlb
import json
from pyclustering.utils import timedcall
import sys
import os
from datetime import datetime
from Algorithm import DBSCAN, KMEANS, PAM, CLARANS

# Construct Output directory path relative to parent directory
OUTPUT_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Output')

# Load data directly
DATASET_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'datasets', 'marketing_campaign.csv')

# Configuration
K_CLUSTERS = 4
FEATURES = ['Income', 'Age', 'Total_Spend']

# CLARANS parameters
CLARANS_NUM_LOCAL = 5
CLARANS_MAX_NEIGHBORS = 100

# PAM parameters
PAM_CLUSTERS = K_CLUSTERS

# K-Means parameters  
KMEANS_CLUSTERS = K_CLUSTERS
KMEANS_MAX_ITERS = 300

# DBSCAN parameters
DBSCAN_EPS = 0.25
DBSCAN_MIN_SAMPLES = 10


def load_data():
    """Load and prepare data"""
    # Read raw data
    data = pd.read_csv(DATASET_PATH)
    
    # Data preprocessing (from data_processing.py)
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
    
    print(f"✓ Data loaded: {selected_data.shape}")
    print(f"  Features: {FEATURES}\n")
    return selected_data


def train_kmeans(data):
    """Train K-Means model"""
    print("🔵 Training K-Means...", end=" ", flush=True)
    try:
        model = KMEANS(data, n_clusters=KMEANS_CLUSTERS, max_iters=KMEANS_MAX_ITERS)
        ticks, _ = timedcall(model.fit)
        jlb.dump(model, os.path.join(OUTPUT_DIR, 'kMeans.mdl'))
        print(f"✓ {ticks:.4f}s")
        return ticks
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def train_pam(data):
    """Train PAM model"""
    print("Training PAM...", end=" ", flush=True)
    try:
        model = PAM(data, n_clusters=PAM_CLUSTERS)
        ticks, _ = timedcall(model.fit)
        jlb.dump(model, os.path.join(OUTPUT_DIR, 'PAM.mdl'))
        print(f"✓ {ticks:.4f}s")
        return ticks
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def train_clarans(data):
    """Train CLARANS model - CRITICAL FOR COMPARISON"""
    print("Training CLARANS...", end=" ", flush=True)
    try:
        model = CLARANS(
            data, 
            n_clusters=K_CLUSTERS,
            num_local=CLARANS_NUM_LOCAL,
            max_neighbors=CLARANS_MAX_NEIGHBORS
        )
        ticks, _ = timedcall(model.fit)
        jlb.dump(model, os.path.join(OUTPUT_DIR, 'clarans.mdl'))
        print(f"✓ {ticks:.4f}s")
        return ticks
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def train_dbscan(data):
    """Train DBSCAN model"""
    print("Training DBSCAN...", end=" ", flush=True)
    try:
        # DBSCAN needs data in format suitable for distance calculation
        X = data.values
        model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        ticks, _ = timedcall(model.fit, X)
        jlb.dump(model, os.path.join(OUTPUT_DIR, 'DBSCAN.mdl'))
        print(f"✓ {ticks:.4f}s")
        return ticks
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def save_runtime_metadata(runtimes):
    """Save runtime metadata to JSON file for reproducibility"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'runtimes': runtimes,
        'configuration': {
            'k_clusters': K_CLUSTERS,
            'features': FEATURES,
            'clarans_num_local': CLARANS_NUM_LOCAL,
            'clarans_max_neighbors': CLARANS_MAX_NEIGHBORS
        }
    }
    metadata_path = os.path.join(OUTPUT_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    return metadata_path


def verify_models():
    """Verify all models can be loaded and have required methods"""
    print("\n" + "=" * 70)
    print("VERIFYING MODELS")
    print("=" * 70)
    
    models_to_check = [
        ('K-Means', os.path.join(OUTPUT_DIR, 'kMeans.mdl')),
        ('PAM', os.path.join(OUTPUT_DIR, 'PAM.mdl')),
        ('CLARANS', os.path.join(OUTPUT_DIR, 'clarans.mdl')),
        ('DBSCAN', os.path.join(OUTPUT_DIR, 'DBSCAN.mdl')),
    ]
    
    all_ok = True
    
    for name, path in models_to_check:
        try:
            model = jlb.load(path)
            
            # Check required methods
            if name == 'DBSCAN':
                # DBSCAN uses labels_ attribute, not get_labels()
                _ = model.labels_
            else:
                _ = model.get_labels()
                if name in ['PAM', 'CLARANS']:
                    _ = model.get_medoids()
                else:
                    _ = model.get_centroids()
            
            print(f"  ✓ {name:10s} loaded successfully")
            
        except Exception as e:
            print(f"  ✗ {name:10s} ERROR: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main execution pipeline"""
    
    print("\n" + "=" * 70)
    print("RETRAINING ALL CLUSTERING MODELS (k=4)")
    print("=" * 70 + "\n")
    
    # Configuration display
    print("CONFIGURATION:")
    print(f"  • Number of clusters (k): {K_CLUSTERS}")
    print(f"  • Features: {FEATURES}")
    print(f"  • CLARANS params: num_local={CLARANS_NUM_LOCAL}, max_neighbors={CLARANS_MAX_NEIGHBORS}")
    print()
    
    # Load data
    data = load_data()
    
    # Train models
    print("TRAINING MODELS")
    print("-" * 70)
    
    runtimes = {}
    runtimes['K-Means'] = train_kmeans(data)
    runtimes['PAM'] = train_pam(data)
    runtimes['CLARANS'] = train_clarans(data)  # CRITICAL: CLARANS must be trained
    runtimes['DBSCAN'] = train_dbscan(data)
    
    # Save runtime metadata
    save_runtime_metadata({k: v for k, v in runtimes.items() if v is not None})
    
    # Verify
    all_ok = verify_models()
    
    print("\n" + "=" * 70)
    if all_ok:
        print("ALL MODELS TRAINED AND VERIFIED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNEXT STEP:")
        print("   Run: python compare_algorithms.py")
        print("   This will generate comprehensive comparison with CLARANS included.")
    else:
        print("SOME MODELS FAILED VERIFICATION!")
        print("=" * 70)
    
    # Print runtime summary
    print("\nRUNTIME SUMMARY:")
    print("-" * 70)
    for algo, time in runtimes.items():
        if time is not None:
            print(f"  {algo:10s}: {time:.4f}s")
        else:
            print(f"  {algo:10s}: FAILED")


if __name__ == '__main__':
    main()
