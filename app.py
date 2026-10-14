from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import io

app = Flask(__name__)
app.secret_key = "wine-clustering-demo-key"

def analyze_wine(n_clusters=3):
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    scaled = StandardScaler().fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(scaled)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hi_labels = hierarchical.fit_predict(scaled)

    pca = PCA(n_components=2)
    points = pca.fit_transform(scaled)

    def metrics(labels):
        return {
            "silhouette": round(float(silhouette_score(scaled, labels)), 4),
            "calinski": round(float(calinski_harabasz_score(scaled, labels)), 2),
            "davies": round(float(davies_bouldin_score(scaled, labels)), 4),
        }

    result = df.copy()
    result["KMeans_Cluster"] = km_labels
    result["Hierarchical_Cluster"] = hi_labels

    km_counts = pd.Series(km_labels).value_counts().sort_index()
    hi_counts = pd.Series(hi_labels).value_counts().sort_index()

    return {
        "rows": int(df.shape[0]),
        "features": int(df.shape[1]),
        "feature_names": list(df.columns),
        "sample_rows": df.head(6).round(3).to_dict(orient="records"),
        "sample_columns": list(df.columns),
        "km_metrics": metrics(km_labels),
        "hi_metrics": metrics(hi_labels),
        "km_counts": [int(km_counts.get(i, 0)) for i in range(n_clusters)],
        "hi_counts": [int(hi_counts.get(i, 0)) for i in range(n_clusters)],
        "pca_x": [round(float(x), 4) for x in points[:, 0]],
        "pca_y": [round(float(y), 4) for y in points[:, 1]],
        "km_labels": [int(x) for x in km_labels],
        "hi_labels": [int(x) for x in hi_labels],
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "n_clusters": n_clusters,
        "csv": result.to_csv(index=False),
        "comparison": "K-Means" if metrics(km_labels)["silhouette"] > metrics(hi_labels)["silhouette"] else ("Hierarchical Clustering" if metrics(hi_labels)["silhouette"] > metrics(km_labels)["silhouette"] else "Both algorithms"),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    n_clusters = 3
    if request.method == "POST":
        try:
            n_clusters = int(request.form.get("clusters", "3"))
            if not 2 <= n_clusters <= 8:
                raise ValueError
        except ValueError:
            flash("Choose a cluster count between 2 and 8.")
            return redirect(url_for("index"))
    data = analyze_wine(n_clusters)
    return render_template("index.html", data=data)

@app.route("/download")
def download():
    try:
        n_clusters = int(request.args.get("clusters", "3"))
        if not 2 <= n_clusters <= 8:
            n_clusters = 3
    except ValueError:
        n_clusters = 3
    data = analyze_wine(n_clusters)
    buffer = io.BytesIO(data["csv"].encode("utf-8"))
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name="wine_clustering_results.csv")

if __name__ == "__main__":
    app.run(debug=True)
