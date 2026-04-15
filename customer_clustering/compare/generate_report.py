"""
Academic Clustering Algorithm Comparison Report Generator

This module generates a neutral, scientific HTML report comparing different
clustering algorithms based on multiple evaluation metrics.

Metrics used:
- Silhouette Score (higher is better, range: -1 to 1)
- Davies-Bouldin Index (lower is better, range: 0 to inf)
- Calinski-Harabasz Score (higher is better, range: 0 to inf)
- Clustering Cost (lower is better, sum of distances to nearest center)
- Execution Time (lower is better)
"""

import json
import pandas as pd
from datetime import datetime
import os

# Construct Output directory path relative to this file
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Output')


def load_metrics():
    """Load metrics from CSV file"""
    try:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, 'algorithm_comparison.csv'))
        return df
    except FileNotFoundError:
        print("✗ algorithm_comparison.csv not found!")
        print("   Run: python compare_algorithms.py")
        return None


def load_css():
    """Load CSS content from style.css"""
    try:
        with open('style.css', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def generate_html(df, css_content=''):
    """Generate neutral, academic HTML report from metrics DataFrame"""
    
    # Prepare data
    algorithms = df['Algorithm'].tolist()
    n_algos = len(algorithms)
    
    # Format data for display and JSON
    silhouette_data = [None if pd.isna(x) else round(float(x), 4) for x in df['Silhouette'].tolist()]
    davies_bouldin_data = [None if pd.isna(x) else round(float(x), 4) for x in df['Davies-Bouldin'].tolist()]
    calinski_harabasz_data = [None if pd.isna(x) else round(float(x), 1) for x in df['Calinski-Harabasz'].tolist()]
    runtime_data = [round(float(x), 4) for x in df['Time (s)'].tolist()]
    
    # Convert to JSON for charts
    algorithms_json = json.dumps(algorithms)
    silhouette_json = json.dumps(silhouette_data)
    davies_json = json.dumps(davies_bouldin_data)
    ch_json = json.dumps(calinski_harabasz_data)
    runtime_json = json.dumps(runtime_data)
    
    # Neutral colors (blue gradient)
    colors = ['rgba(52, 152, 219, 0.8)' for _ in algorithms]
    colors_json = json.dumps(colors)
    
    # Get timestamp
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clustering Algorithms Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="../compare/style.css">
</head>
<body>
    <div class="header">
        <h1>Clustering Algorithms Comparison Report</h1>
        <p>Objective evaluation of clustering algorithms on customer segmentation data</p>
    </div>

    <div class="container">
        <div class="content">
            <!-- EXECUTIVE SUMMARY -->
            <div class="section">
                <h2 class="section-title">Executive Summary</h2>
                <p>
                    This report presents a comprehensive, unbiased comparison of multiple clustering algorithms
                    applied to a customer segmentation dataset. The evaluation uses four well-established metrics
                    that measure different aspects of clustering quality and efficiency.
                </p>
                <p style="margin-top: 15px;">
                    <strong>Algorithms Evaluated:</strong> """ + ", ".join(algorithms) + """
                </p>
                <p style="margin-top: 10px;">
                    <strong>Dataset:</strong> Multiple clustering solutions identified across """ + str(n_algos) + """ different algorithms
                </p>
            </div>

            <!-- METRICS TABLE -->
            <div class="section">
                <h2 class="section-title">Performance Metrics Summary</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Silhouette<br/>(↑ Higher)</th>
                            <th>Davies-Bouldin<br/>(↓ Lower)</th>
                            <th>Calinski-Harabasz<br/>(↑ Higher)</th>
                            <th>Time (s)<br/>(↓ Lower)</th>
                            <th>Clusters</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add table rows
    for idx, row in df.iterrows():
        silhouette_val = f"{row['Silhouette']:.4f}" if pd.notna(row['Silhouette']) else 'N/A'
        davies_val = f"{row['Davies-Bouldin']:.4f}" if pd.notna(row['Davies-Bouldin']) else 'N/A'
        ch_val = f"{row['Calinski-Harabasz']:.1f}" if pd.notna(row['Calinski-Harabasz']) else 'N/A'
        time_val = f"{row['Time (s)']:.4f}" if pd.notna(row['Time (s)']) else 'N/A'
        
        html += f"""                        <tr>
                            <td><strong>{row['Algorithm']}</strong></td>
                            <td>{silhouette_val}</td>
                            <td>{davies_val}</td>
                            <td>{ch_val}</td>
                            <td>{time_val}</td>
                            <td>{int(row['Clusters'])}</td>
                        </tr>
"""
    
    html += """                    </tbody>
                </table>
            </div>

            <!-- METRIC DEFINITIONS -->
            <div class="section">
                <h2 class="section-title">Metric Definitions</h2>
                
                <h3 class="subsection-title">Silhouette Score</h3>
                <p>
                    <strong>Range:</strong> -1 to 1 | <strong>Better:</strong> Higher values<br/>
                    Measures how similar an object is to its own cluster compared to other clusters.
                    A value of 1 indicates perfect clustering, 0 indicates overlapping clusters, 
                    and negative values indicate poorly clustered objects.
                </p>
                
                <h3 class="subsection-title">Davies-Bouldin Index</h3>
                <p>
                    <strong>Range:</strong> 0 to ∞ | <strong>Better:</strong> Lower values<br/>
                    Represents the average similarity between each cluster and its most similar cluster,
                    where similarity is the ratio of within-cluster distances to between-cluster distances.
                    Lower values indicate better-separated clusters.
                </p>
                
                <h3 class="subsection-title">Calinski-Harabasz Score</h3>
                <p>
                    <strong>Range:</strong> 0 to ∞ | <strong>Better:</strong> Higher values<br/>
                    The ratio of between-cluster variance to within-cluster variance. Higher values indicate
                    denser and better-separated clusters. Also known as the Variance Ratio Criterion.
                </p>
                
                <h3 class="subsection-title">Execution Time</h3>
                <p>
                    <strong>Unit:</strong> Seconds | <strong>Better:</strong> Lower values<br/>
                    The actual computational time required to train the clustering model on the dataset.
                    Important for practical applications with time constraints.
                </p>
            </div>

            <!-- VISUAL COMPARISON -->
            <div class="section">
                <h2 class="section-title">Visual Comparison</h2>
                
                <h3 class="subsection-title">Silhouette Score Comparison</h3>
                <div class="chart-container">
                    <canvas id="silhouetteChart"></canvas>
                </div>

                <h3 class="subsection-title">Davies-Bouldin Index Comparison</h3>
                <div class="chart-container">
                    <canvas id="daviesChart"></canvas>
                </div>

                <h3 class="subsection-title">Calinski-Harabasz Score Comparison</h3>
                <div class="chart-container">
                    <canvas id="chChart"></canvas>
                </div>

                <h3 class="subsection-title">Execution Time Comparison</h3>
                <div class="chart-container">
                    <canvas id="timeChart"></canvas>
                </div>
            </div>

            <!-- ALGORITHM CHARACTERISTICS -->
            <div class="section">
                <h2 class="section-title">Algorithm Characteristics</h2>
                
                <h3 class="subsection-title">Algorithm Types</h3>
                <div class="comparison-grid">
                    <div class="comparison-box">
                        <h4>Medoid-Based Algorithms</h4>
                        <p>Use actual data points as cluster centers.</p>
                        <ul class="feature-list">
                            <li>PAM (Partitioning Around Medoids)</li>
                            <li>CLARANS (Clustering Large Applications based on Randomized Search)</li>
                        </ul>
                        <p style="margin-top: 15px;">
                            <strong>Advantages:</strong> Results are interpretable (centers are real data points),
                            robust to outliers.
                        </p>
                    </div>
                    
                    <div class="comparison-box">
                        <h4>Centroid-Based Algorithms</h4>
                        <p>Compute mathematical centroids (averages) as cluster centers.</p>
                        <ul class="feature-list">
                            <li>K-Means</li>
                        </ul>
                        <p style="margin-top: 15px;">
                            <strong>Advantages:</strong> Computationally efficient, scalable to large datasets.
                        </p>
                    </div>

                    <div class="comparison-box">
                        <h4>Density-Based Algorithms</h4>
                        <p>Group objects that are closely packed together, separated by regions of low density.</p>
                        <ul class="feature-list">
                            <li>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</li>
                        </ul>
                        <p style="margin-top: 15px;">
                            <strong>Advantages:</strong> Can find arbitrary-shaped clusters, automatically detects outliers/noise,
                            does not require specifying number of clusters in advance.
                        </p>
                    </div>
                </div>
            </div>

            <!-- ANALYSIS AND INTERPRETATION -->
            <div class="section">
                <h2 class="section-title">Analysis and Interpretation</h2>
                
                <h3 class="subsection-title">Key Findings</h3>
                
                <div class="note-box">
                    <strong>Note:</strong> This analysis presents objective metrics only. Algorithm selection 
                    should depend on specific application requirements, including interpretability needs, 
                    computational constraints, and domain-specific considerations.
                </div>
                
                <p style="margin-top: 20px;">
                    The comparison reveals differences in how various algorithms cluster the customer segmentation data:
                </p>
                
                <ul class="feature-list" style="margin-top: 15px;">
                    <li><strong>Cluster Quality:</strong> Measured by Silhouette Score and Davies-Bouldin Index</li>
                    <li><strong>Variance Separation:</strong> Measured by Calinski-Harabasz Score</li>
                    <li><strong>Computational Efficiency:</strong> Measured by Execution Time</li>
                </ul>
                
                <h3 class="subsection-title">Considerations for Algorithm Selection</h3>
                <div class="comparison-box">
                    <h4>Performance Priority</h4>
                    <div class="comparison-item">
                        Choose the algorithm with highest Silhouette Score for best cluster cohesion and separation.
                    </div>
                </div>
                
                <div class="comparison-box">
                    <h4>Interpretability Priority</h4>
                    <div class="comparison-item">
                        Choose a medoid-based algorithm (PAM, CLARANS) if cluster centers must be actual data points.
                    </div>
                </div>
                
                <div class="comparison-box">
                    <h4>Scalability Priority</h4>
                    <div class="comparison-item">
                        Choose centroid-based algorithms (K-Means) for computational efficiency on larger datasets.
                    </div>
                </div>

                <div class="comparison-box">
                    <h4>Automatic Cluster Detection</h4>
                    <div class="comparison-item">
                        Choose DBSCAN if you need the algorithm to automatically determine the number of clusters
                        and handle outliers/noise points without manual parameter tuning for k.
                    </div>
                </div>
            </div>

            <!-- RECOMMENDATIONS -->
            <div class="section">
                <h2 class="section-title">Recommendations</h2>
                <p>
                    Based on the metrics presented in this report, consider the following recommendations
                    when selecting a clustering algorithm for your specific use case:
                </p>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Best Quality</div>
                        <div class="metric-value" id="best-quality">--</div>
                        <div class="metric-direction">Highest Silhouette Score</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Best Separation</div>
                        <div class="metric-value" id="best-separation">--</div>
                        <div class="metric-direction">Highest Calinski-Harabasz Score</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Fastest</div>
                        <div class="metric-value" id="best-speed">--</div>
                        <div class="metric-direction">Lowest Execution Time</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p><strong>Report Generated:</strong> """ + report_time + """</p>
            <p><strong>Data Source:</strong> algorithm_comparison.csv</p>
            <p>This is a neutral, academic comparison of clustering algorithms.</p>
        </div>
    </div>

    <script>
        // Chart data
        const algorithms = """ + algorithms_json + """;
        const silhouetteData = """ + silhouette_json + """;
        const daviesData = """ + davies_json + """;
        const chData = """ + ch_json + """;
        const runtimeData = """ + runtime_json + """;
        const colors = """ + colors_json + """;

        const chartConfig = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        font: { size: 11 }
                    }
                },
                x: {
                    ticks: {
                        font: { size: 11 }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: { size: 12 },
                    bodyFont: { size: 11 }
                }
            }
        };

        // Silhouette Chart
        new Chart(document.getElementById('silhouetteChart'), {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [{
                    label: 'Silhouette Score',
                    data: silhouetteData,
                    backgroundColor: colors,
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1,
                }]
            },
            options: chartConfig
        });

        // Davies-Bouldin Chart
        new Chart(document.getElementById('daviesChart'), {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [{
                    label: 'Davies-Bouldin Index',
                    data: daviesData,
                    backgroundColor: colors,
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1,
                }]
            },
            options: chartConfig
        });

        // Calinski-Harabasz Chart
        new Chart(document.getElementById('chChart'), {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [{
                    label: 'Calinski-Harabasz Score',
                    data: chData,
                    backgroundColor: colors,
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1,
                }]
            },
            options: chartConfig
        });

        // Time Chart
        new Chart(document.getElementById('timeChart'), {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [{
                    label: 'Execution Time (seconds)',
                    data: runtimeData,
                    backgroundColor: colors,
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1,
                }]
            },
            options: chartConfig
        });

        // Find and display best algorithms
        function findBestAlgorithm(data, ascending = false) {
            let bestIdx = -1;
            let bestVal = ascending ? Infinity : -Infinity;
            
            for (let i = 0; i < data.length; i++) {
                if (data[i] !== null) {
                    if (ascending ? data[i] < bestVal : data[i] > bestVal) {
                        bestVal = data[i];
                        bestIdx = i;
                    }
                }
            }
            
            return bestIdx >= 0 ? algorithms[bestIdx] : '--';
        }

        document.getElementById('best-quality').textContent = findBestAlgorithm(silhouetteData);
        document.getElementById('best-separation').textContent = findBestAlgorithm(chData);
        document.getElementById('best-speed').textContent = findBestAlgorithm(runtimeData, true);
    </script>
</body>
</html>
"""
    
    return html


def main():
    """Main execution"""
    print("Generating academic HTML report from comparison results...\n")
    
    # Load CSS
    css_content = load_css()
    
    # Load metrics
    df = load_metrics()
    if df is None:
        return
    
    # Generate HTML
    html_content = generate_html(df, css_content)
    
    # Save HTML
    output_file = os.path.join(OUTPUT_DIR, 'clustering_comparison_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report generated: {output_file}")
    print(f"\nReport includes:")
    print(f"   • Neutral, academic perspective")
    print(f"   • Comprehensive metrics table")
    print(f"   • 4 interactive charts (Silhouette, Davies-Bouldin, Calinski-Harabasz, Time)")
    print(f"   • Metric definitions and interpretations")
    print(f"   • Algorithm characteristics and comparison")
    print(f"   • Evidence-based recommendations")
    print(f"\nOpen in browser: {output_file}")


if __name__ == '__main__':
    main()
