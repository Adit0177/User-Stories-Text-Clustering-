from flask import Flask, request, render_template,url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from wordcloud import WordCloud
from flask import Flask, render_template, redirect, url_for


app = Flask(__name__)

def perform_clustering(df):
    # Convert text data to vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text_column'])

    # Perform KMeans clustering
    k = 3  # Number of clusters, can be changed as needed
    kmeans = KMeans(n_clusters=k)
    df['cluster'] = kmeans.fit_predict(X)

    # Generate word clouds for each cluster
    cluster_wordclouds = {}
    for cluster, data in df.groupby('cluster'):
        text = " ".join(data['text_column'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Convert wordcloud to PNG format
        buffer = BytesIO()
        wordcloud.to_image().save(buffer, format="PNG")

        # Convert PNG image to base64 string
        wordcloud_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        cluster_wordclouds[cluster] = wordcloud_base64

    return df['cluster'], cluster_wordclouds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df.columns=["text_column"]
            clusters, cluster_wordclouds = perform_clustering(df)
            return render_template('results.html',  cluster_wordclouds=cluster_wordclouds)

    return render_template('index.html')

# def perform_clustering(df):
#     # Convert text data to vectors using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(df['text_column'])

#     # Perform KMeans clustering
#     k = 3  # Number of clusters, can be changed as needed
#     kmeans = KMeans(n_clusters=k)
#     df['cluster'] = kmeans.fit_predict(X)

#     # Get terms associated with each centroid
#     feature_names = vectorizer.get_feature_names_out()
#     cluster_terms = {}
#     for i in range(k):
#         centroid = kmeans.cluster_centers_[i]
#         top_features = [feature_names[ind] for ind in centroid.argsort()[:-11:-1]]  # Get top 10 terms
#         cluster_terms[i] = top_features

#     # Create bar charts for clusters
#     plt.figure(figsize=(10, 6))
#     cluster_counts = df['cluster'].value_counts().sort_index()
#     cluster_counts.plot(kind='bar', color='skyblue')
#     plt.title('Cluster Distribution')
#     plt.xlabel('Cluster')
#     plt.ylabel('Count')

#     # Save the bar chart to a bytes object
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     bar_chart = base64.b64encode(buffer.getvalue())
#     plt.close()

#     return df['cluster'], bar_chart, cluster_terms


# ... [rest of the imports]





if __name__ == '__main__':
    app.run(debug=True)