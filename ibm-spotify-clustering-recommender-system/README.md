## Spotify Clustering and Recommender System
This project explores unsupervised learning techniques to cluster and recommend songs from the Spotify API.

The goal of this project is to develop a song recommender system using unsupervised learning techniques. The Spotify API is used to collect data on songs, and this data is then preprocessed and clustered using K-Means, Agglomerative Clustering and Mean Shift.

The resulting clusters from the first two models are then used to create a song recommender using Euclidian distance and cosine similarity.

### Data
The data used in this project was collected from the Spotify API. The data includes information on songs such as their name, artist, and audio features such as danceability and energy.

### Models
- K-Means
The number of clusters is determined using the Elbow method, Silhouette score and David Bouldin index. 
- Agglomerative Clustering 
The number of clusters is determined using the hierarchical dendrogram.
- Mean Shift
The number of clusters is determined by using the estimate_bandwidth function.

### Report
A report on this project can be found in the report folder. This report includes information on the data, models, and results of the project.

### Files
#### The following files are included in this project:

- csvs: folder containing the csv files used in the project
- models: folder containing the trained machine learning models
- report: folder containing the project report
- retained_features.json: file containing the audio features retained for clustering
- spotify_api.ipynb: Jupyter notebook used to collect data from the Spotify API
- spotify_clustering.ipynb: Jupyter notebook used to preprocess and cluster the data
- spotify_recommender.ipynb: Jupyter notebook used to create the song recommender

### Instructions
#### To run this project, follow these steps:

- Clone the repository
- Collect data from the Spotify API using the spotify_api.ipynb notebook
- Preprocess and cluster the data using the spotify_clustering.ipynb notebook
- Create a song recommender using the spotify_recommender.ipynb notebook
Note that you will need a Spotify API client ID and secret key to collect data from the Spotify API.
