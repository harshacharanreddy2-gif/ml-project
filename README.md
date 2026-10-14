# Wine Sample Clustering Website

A local Flask dashboard based on the notebook's Wine dataset workflow. It compares K-Means and Agglomerative (Hierarchical) Clustering using standardized features, PCA visualizations, cluster counts, and Silhouette / Calinski-Harabasz / Davies-Bouldin metrics.

## Run in VS Code (Windows)

1. Extract this ZIP.
2. Open the extracted `wine_clustering_website` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked, use Command Prompt or run `.venv\Scripts\python.exe` directly.

5. Install libraries:

   ```powershell
   pip install -r requirements.txt
   ```

6. Start the website:

   ```powershell
   python app.py
   ```

7. Open this address in your browser:

   http://127.0.0.1:5000

## Features

- K-Means and Hierarchical Clustering on scikit-learn's 178-sample, 13-feature Wine dataset
- Select 2–8 clusters and rerun the analysis
- PCA scatter plots for both algorithms
- Cluster distribution and metric comparison charts
- Silhouette, Calinski-Harabasz and Davies-Bouldin scores
- Download results as `wine_clustering_results.csv`

## Notes

- Internet is required for the Chart.js script loaded from its CDN.
- This is a local demo website; it does not upload data to a server.
- The project uses the same key steps as your notebook, but the website computes results when the page is opened.
