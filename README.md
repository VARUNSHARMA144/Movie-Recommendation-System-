# Movie Recommendation System

This project involves building a comprehensive movie recommendation system using the MovieLens dataset, incorporating various collaborative filtering and content-based approaches. The system aims to provide personalized movie suggestions to users.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Installation and Setup](#installation-and-setup)
4.  [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
5.  [Recommendation Algorithms Implemented](#recommendation-algorithms-implemented)
6.  [Model Evaluation](#model-evaluation)
7.  [Usage](#usage)
8.  [Future Improvements](#future-improvements)
9.  [Contributing](#contributing)
10. [License](#license)

## Project Overview

This project focuses on developing a movie recommendation system that helps users discover content relevant to their interests from a vast array of options. [cite: 54] We implemented and evaluated several recommendation algorithms, including:

* **Collaborative Filtering (User-Based and Item-Based):** Recommends items based on user-item interactions. [cite: 52]
* **Matrix Factorization (SVD):** Identifies latent factors from user-item interactions. [cite: 74, 75]
* **Content-Based Filtering:** Recommends items based on their attributes and a user's past preferences. [cite: 52]
* **Hybrid Recommendation System:** Combines the strengths of multiple approaches. [cite: 81]

The goal is to provide personalized movie recommendations by analyzing user behavior and movie attributes. [cite: 55]

## Dataset

The project utilizes the **MovieLens latest-small dataset**, a widely-used benchmark dataset for recommendation systems research. [cite: 53, 56]

The dataset consists of the following files:
* `movies.csv`: Contains movie IDs, titles, and genres. [cite: 57]
* `ratings.csv`: Contains user ratings for movies, including UserID, MovieID, Rating, and Timestamp. [cite: 57]
* `links.csv`: Provides MovieID, IMDB ID, and TMDB ID mappings. [cite: 57]
* `tags.csv`: Contains user-generated tags for movies, along with UserID, MovieID, Tag, and Timestamp. [cite: 57]

The dataset is downloaded directly within the notebook. [cite: 57]

## Installation and Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd movie-recommendation-system
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```
    (These are the libraries explicitly imported in the provided code. [cite: 57])

## Data Preprocessing and Feature Engineering

The data underwent a thorough cleaning and feature engineering process to prepare it for the recommendation algorithms:

* **Missing Value Handling:** Checked for and addressed missing values (e.g., `tmdbId` in `links.csv` was filled with -1). [cite: 59]
* **Duplicate Removal:** Removed duplicate entries from `movies` and `ratings` datasets. [cite: 59]
* **Timestamp Conversion:** Converted timestamp columns to datetime objects. [cite: 59]
* **Feature Extraction:** [cite: 60]
    * Extracted `year` from movie titles.
    * Created one-hot encoded genre features.
    * Generated user activity features (`total_ratings`, `avg_rating`, `rating_std` per user).
    * Generated movie popularity features (`total_ratings`, `avg_rating`, `rating_std` per movie).
    * Extracted time-based features (year, month, day, day of week) from ratings timestamps.
* **Data Integrity Checks:** Ensured consistent movie IDs, valid rating ranges (0.5-5.0), and handled cold-start issues by identifying users/movies with very few ratings. [cite: 61, 62]
* **Outlier Handling:** Removed users with extreme rating patterns (e.g., rating everything the same). [cite: 66]
* **Data Transformation:** Normalized ratings by user mean and applied Min-Max scaling to dense features. [cite: 66]
* **User-Item Matrix:** Created a pivoted user-item matrix for collaborative filtering. [cite: 66]

## Recommendation Algorithms Implemented

The project implements and evaluates five different recommendation algorithms:

1.  **User-Based Collaborative Filtering:** Identifies users with similar tastes and recommends movies liked by those similar users but not yet watched by the target user. [cite: 69]
2.  **Item-Based Collaborative Filtering:** Recommends movies similar to those the user has already liked, based on item-to-item similarity. [cite: 72]
3.  **Matrix Factorization (SVD):** Uses Singular Value Decomposition to uncover latent factors that explain user-item interactions and predict ratings for unrated movies. [cite: 75]
4.  **Content-Based Filtering:** Builds a profile of the user's preferences based on the genres of movies they have rated highly, then recommends movies with similar genre attributes. [cite: 78]
5.  **Hybrid Recommendation System:** Combines the predictions from the user-based, item-based, SVD, and content-based methods by normalizing and weighting their scores to provide a more robust recommendation. [cite: 81]

## Model Evaluation

The algorithms are evaluated using a test set of users to assess their performance. [cite: 84]

**Metrics Used:**
* **RMSE (Root Mean Squared Error):** Measures the accuracy of predicted ratings. Lower RMSE indicates better accuracy. [cite: 87]
* **Precision:** The proportion of recommended items that are actually relevant to the user. [cite: 87]
* **Recall:** The proportion of relevant items that are successfully recommended to the user. [cite: 87]
* **F1-Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both. [cite: 88]

The evaluation process involves:
1.  Selecting a subset of users for testing. [cite: 84]
2.  Splitting their ratings into training and test sets. [cite: 85]
3.  Generating recommendations using each algorithm based on the training data. [cite: 86]
4.  Calculating RMSE, Precision, and Recall by comparing predicted recommendations against the held-out test set. [cite: 86, 87]

Comparison plots for each metric are generated to visualize the performance of different algorithms. [cite: 88, 89]

## Usage

The `MovieRecommendationSystem` class provides a unified interface for generating recommendations.

```python
import pandas as pd
# Assuming 'movies' and 'ratings' DataFrames are loaded as per the project
# from your data loading script or a pre-processed file.

# Load the dataset (example, replace with your actual loading logic if different)
# !wget [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
# !unzip ml-latest-small.zip
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Preprocessing steps for movies and ratings (as done in the project)
# For example, extracting year and one-hot encoding genres for 'movies'
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['title'] = movies['title'].str.replace(r' \(\d{4}\)$', '', regex=True)
genres = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# You'd also merge user activity and movie popularity features here
# For demonstration purposes, let's assume `ratings` and `movies` are ready

from movie_recommendation_system_script import MovieRecommendationSystem, user_based_recommendation, item_based_recommendation, svd_recommendation, content_based_recommendation, hybrid_recommendation

# Initialize the recommender system
recommender = MovieRecommendationSystem(ratings, movies)

# Example 1: Get recommendations for a specific user
user_id = 1
recommendations = recommender.recommend_for_user(user_id, method='hybrid')
print(f"Recommendations for user {user_id} using Hybrid method:")
print(recommendations)

# Example 2: Find similar movies to a given movie
movie_id = 1  # Example movie ID (Toy Story)
similar_movies = recommender.recommend_similar_movies(movie_id)
print(f"\nMovies similar to {movies[movies['movieId'] == movie_id]['title'].values[0]}:")
print(similar_movies)

# Example 3: Get popular movies by rating
popular_by_rating = recommender.popular_movies(by='rating')
print("\nPopular movies by rating:")
print(popular_by_rating)

# Example 4: Get recommendations for a new user based on genre preferences
genre_preferences = {
    'Action': 5,
    'Adventure': 4,
    'Sci-Fi': 5,
    'Drama': 2,
    'Comedy': 3
}
new_user_recs = recommender.recommend_for_new_user(genre_preferences)
print("\nRecommendations for new user with genre preferences:")
print(new_user_recs)
