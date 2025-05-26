import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Dataset Loading ---
# MovieLens dataset can be downloaded from: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# Make sure to unzip it into a folder named 'ml-latest-small' in the same directory as your script.

try:
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    # links = pd.read_csv('ml-latest-small/links.csv') # Not directly used in core logic provided, but good to have
    # tags = pd.read_csv('ml-latest-small/tags.csv') # Not directly used in core logic provided, but good to have
except FileNotFoundError:
    print("Error: MovieLens dataset not found. Please download 'ml-latest-small.zip' from")
    print("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    print("and extract it to a folder named 'ml-latest-small' in the same directory as this script.")
    exit()

print("Datasets loaded successfully.")

# --- 2. Data Cleaning and Preprocessing (based on PDF Page 3) ---

# Convert timestamp to datetime objects for easier analysis
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Check for missing values in core datasets
# For 'movies' and 'ratings', usually clean. Check for 'links.csv' for example.
# movies.isnull().sum()
# ratings.isnull().sum()
# links.isnull().sum() # Assuming links.csv is loaded
# Example of filling missing tmdbId (if using links.csv)
# if 'tmdbId' in links.columns:
#     links['tmdbId'] = links['tmdbId'].fillna(-1).astype(int)

# Check for and remove duplicate entries
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

print("Data cleaning and preprocessing complete.")

# --- 3. Feature Selection and Engineering (based on PDF Page 4) ---

# Extract year from movie titles
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
# Clean movie titles by removing the year
movies['title'] = movies['title'].str.replace(r' \(\d{4}\)$', '', regex=True)

# One-hot encode genres
genres_dummies = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genres_dummies], axis=1)
# Drop the original 'genres' column if not needed directly
# movies.drop('genres', axis=1, inplace=True)

# Create user activity features
user_activity_stats = ratings.groupby('userId').agg(
    total_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    rating_std=('rating', 'std')
).reset_index()

# Create movie popularity features
movie_popularity_stats = ratings.groupby('movieId').agg(
    total_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    rating_std=('rating', 'std')
).reset_index()

# Merge these features into the main 'ratings' DataFrame
# IMPORTANT: Renaming columns during merge or before to avoid conflicts and ensure specific names
ratings = pd.merge(ratings, user_activity_stats[['userId', 'total_ratings']], on='userId', how='left', suffixes=('_original', '_user'))
ratings.rename(columns={'total_ratings': 'total_ratings_user'}, inplace=True) # Ensure consistent naming

ratings = pd.merge(ratings, movie_popularity_stats[['movieId', 'total_ratings', 'avg_rating']], on='movieId', how='left', suffixes=('_user_temp', '_movie'))
# After the above merge, the movie's total ratings column will be named 'total_ratings_movie'
# and avg_rating will be 'avg_rating_movie'
ratings.rename(columns={'total_ratings_movie': 'total_ratings_movie', 'avg_rating_movie': 'avg_rating_movie'}, inplace=True)


# Extract time-based features from ratings timestamp (if needed for advanced analysis)
ratings['rating_year'] = ratings['timestamp'].dt.year
ratings['rating_month'] = ratings['timestamp'].dt.month
ratings['rating_day'] = ratings['timestamp'].dt.day
ratings['rating_dayofweek'] = ratings['timestamp'].dt.dayofweek

print("Feature engineering complete.")

# --- 4. Ensuring Data Integrity and Consistency (based on PDF Page 5) ---

# Ensure rating values are within the expected range (0.5-5.0) - typically handled by dataset
# print(ratings['rating'].min(), ratings['rating'].max())

# Identify and optionally handle cold start problem (users/movies with very few ratings)
# For now, we'll just identify. Removal strategy varies.
min_user_ratings = 5
min_movie_ratings = 5

cold_start_users = user_activity_stats[user_activity_stats['total_ratings'] < min_user_ratings]['userId']
cold_start_movies = movie_popularity_stats[movie_popularity_stats['total_ratings'] < min_movie_ratings]['movieId']

print(f"Identified {len(cold_start_users)} cold start users (less than {min_user_ratings} ratings).")
print(f"Identified {len(cold_start_movies)} cold start movies (less than {min_movie_ratings} ratings).")

print("Data integrity checks complete.")

# --- 5. Handling Outliers and Data Transformations (based on PDF Page 8) ---

# Remove extreme outliers (users who rate everything the same, std_dev = 0 or very close to 0)
# A small epsilon to avoid division by zero or floating point issues
epsilon = 1e-6
outlier_users = user_activity_stats[user_activity_stats['rating_std'].fillna(0) < epsilon]['userId']
ratings = ratings[~ratings['userId'].isin(outlier_users)]
print(f"Removed {len(outlier_users)} outlier users (near zero rating standard deviation).")

# Rating Normalization: Normalize ratings by user mean
# (This typically means subtracting user's average rating from their ratings)
user_mean_ratings = ratings.groupby('userId')['rating'].mean().reset_index()
user_mean_ratings.columns = ['userId', 'user_avg_rating']
ratings = pd.merge(ratings, user_mean_ratings, on='userId', how='left')
ratings['rating_norm'] = ratings['rating'] - ratings['user_avg_rating']


# --- FIX START: Ensure 'total_ratings_user_scaled' and 'total_ratings_movie_scaled' exist ---
# Apply Min-Max scaling to dense features like user activity and movie popularity
scaler = MinMaxScaler()

# Scale user activity feature: 'total_ratings_user'
if 'total_ratings_user' in ratings.columns:
    ratings['total_ratings_user_scaled'] = scaler.fit_transform(ratings[['total_ratings_user']])
else:
    print("Warning: 'total_ratings_user' column not found for scaling. Check feature engineering steps.")
    # If it's truly missing, you might need to handle the qcut line later with a check.

# Scale movie popularity feature: 'total_ratings_movie'
if 'total_ratings_movie' in ratings.columns:
    ratings['total_ratings_movie_scaled'] = scaler.fit_transform(ratings[['total_ratings_movie']])
else:
    print("Warning: 'total_ratings_movie' column not found for scaling. Check feature engineering steps.")
    # If it's truly missing, you might need to handle the qcut line later with a check.
# --- FIX END ---


# Create bins for user activity and movie popularity for visualizations (as per PDF Page 10)
if 'total_ratings_user_scaled' in ratings.columns:
    user_activity_bins = pd.qcut(ratings['total_ratings_user_scaled'], q=5, labels=False, duplicates='drop')
    ratings['user_activity_bin'] = user_activity_bins
else:
    print("Cannot create user_activity_bins: 'total_ratings_user_scaled' is missing.")

if 'total_ratings_movie_scaled' in ratings.columns:
    movie_popularity_bins = pd.qcut(ratings['total_ratings_movie_scaled'], q=5, labels=False, duplicates='drop')
    ratings['movie_popularity_bin'] = movie_popularity_bins
else:
    print("Cannot create movie_popularity_bins: 'total_ratings_movie_scaled' is missing.")

print("Outlier handling and data transformations complete.")

# --- 6. User-Item Matrix Creation (for Collaborative Filtering) ---
# Create a pivot table for user-item matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating_norm')
user_movie_matrix_fillna = user_movie_matrix.fillna(0) # Fill NaN with 0 for SVD

print("User-Item matrix created.")

# --- 7. Building Recommendation Algorithms (based on PDF Page 11) ---

class MovieRecommendationSystem:
    def __init__(self, ratings_df, movies_df):
        self.ratings = ratings_df.copy()
        self.movies = movies_df.copy()
        
        # Preprocessing for content-based (ensure genres are ready)
        self.movies_content = self.movies.copy()
        self.genres_list = [col for col in self.movies_content.columns if col not in ['movieId', 'title', 'genres', 'year']]
        
        # Create user-item matrix for collaborative filtering
        self.user_movie_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating_norm').fillna(0)
        self.user_movie_matrix_dense = self.user_movie_matrix.values

        # Perform SVD (latent factors) - done once for efficiency
        try:
            U, sigma, Vt = svds(self.user_movie_matrix_dense, k = min(self.user_movie_matrix_dense.shape)-1) # k = number of latent factors
            sigma = np.diag(sigma)
            self.all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.ratings.pivot_table(index='userId', columns='movieId', values='rating').mean().values
            self.svd_preds_df = pd.DataFrame(self.all_user_predicted_ratings, columns=self.user_movie_matrix.columns, index=self.user_movie_matrix.index)
            print("SVD model trained.")
        except Exception as e:
            self.svd_preds_df = None
            print(f"SVD training failed: {e}. SVD-based recommendations will not be available.")

    def recommend_for_user(self, user_id, method='hybrid', num_recommendations=10):
        if method == 'user_based':
            return self._user_based_recommendation(user_id, num_recommendations)
        elif method == 'item_based':
            return self._item_based_recommendation(user_id, num_recommendations)
        elif method == 'svd':
            return self._svd_recommendation(user_id, num_recommendations)
        elif method == 'content_based':
            return self._content_based_recommendation(user_id, num_recommendations)
        elif method == 'hybrid':
            return self._hybrid_recommendation(user_id, num_recommendations)
        else:
            raise ValueError("Invalid recommendation method. Choose from 'user_based', 'item_based', 'svd', 'content_based', 'hybrid'.")

    def _get_already_watched_movies(self, user_id):
        user_movies = self.ratings[self.ratings['userId'] == user_id]['movieId']
        return user_movies.tolist()

    def _user_based_recommendation(self, user_id, num_recommendations):
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame() # User not found
        
        user_row = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
        # Calculate similarity with all other users
        user_similarities = cosine_similarity(user_row, self.user_movie_matrix).flatten()
        # Exclude self-similarity
        user_similarities[self.user_movie_matrix.index.get_loc(user_id)] = -1 # Set to -1 to ignore self
        
        # Get top similar users
        similar_users_indices = user_similarities.argsort()[-num_recommendations-1:-1][::-1] # Exclude self
        similar_users = self.user_movie_matrix.index[similar_users_indices]

        # Get movies rated by similar users but not by target user
        already_watched = self._get_already_watched_movies(user_id)
        
        recommendations = pd.DataFrame()
        for s_user in similar_users:
            s_user_ratings = self.ratings[(self.ratings['userId'] == s_user) & (~self.ratings['movieId'].isin(already_watched))]
            recommendations = pd.concat([recommendations, s_user_ratings])
        
        # Aggregate and sort by average rating or count of recommendations
        recommendations = recommendations.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        top_movie_ids = recommendations.index.tolist()[:num_recommendations]

        return self.movies[self.movies['movieId'].isin(top_movie_ids)]


    def _item_based_recommendation(self, user_id, num_recommendations):
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame() # User not found
        
        user_ratings_for_items = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings_for_items[user_ratings_for_items > 0].index.tolist()

        if not rated_movies:
            return pd.DataFrame() # No movies rated by this user

        # Calculate item-item similarity (on the movie genre features for content similarity)
        # Or, calculate item-item similarity based on co-occurrence in ratings matrix (more complex)
        # For simplicity, using content-based similarity here as it's common.
        # This requires the 'movies' df to have genre one-hot encodings.
        
        # Calculate item similarity using genres (or any other movie features)
        # Only use genre columns for similarity calculation
        item_features = self.movies_content[self.genres_list].values
        item_similarity_matrix = cosine_similarity(item_features)
        item_similarity_df = pd.DataFrame(item_similarity_matrix, index=self.movies_content['movieId'], columns=self.movies_content['movieId'])

        recommendation_scores = {}
        already_watched = self._get_already_watched_movies(user_id)

        for movie_id in rated_movies:
            if movie_id in item_similarity_df.index: # Check if movie is in similarity matrix
                similar_items = item_similarity_df[movie_id].sort_values(ascending=False)
                # Exclude self and already watched movies
                similar_items = similar_items[similar_items.index != movie_id]
                similar_items = similar_items[~similar_items.index.isin(already_watched)]

                for sim_movie_id, similarity_score in similar_items.items():
                    if sim_movie_id not in recommendation_scores:
                        recommendation_scores[sim_movie_id] = 0
                    # Weight similarity by user's rating for the original item (if available)
                    user_rating = user_ratings_for_items.get(movie_id, 0)
                    recommendation_scores[sim_movie_id] += similarity_score * user_rating

        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        top_movie_ids = [movie_id for movie_id, score in sorted_recommendations[:num_recommendations]]

        return self.movies[self.movies['movieId'].isin(top_movie_ids)]


    def _svd_recommendation(self, user_id, num_recommendations):
        if self.svd_preds_df is None or user_id not in self.svd_preds_df.index:
            return pd.DataFrame() # SVD not available or user not found
        
        user_predicted_ratings = self.svd_preds_df.loc[user_id]
        already_watched = self._get_already_watched_movies(user_id)

        # Filter out movies already watched by the user
        unwatched_movies_predictions = user_predicted_ratings.drop(already_watched, errors='ignore')
        
        # Sort by predicted rating and get top N
        top_movie_ids = unwatched_movies_predictions.sort_values(ascending=False).index.tolist()[:num_recommendations]

        return self.movies[self.movies['movieId'].isin(top_movie_ids)]


    def _content_based_recommendation(self, user_id, num_recommendations):
        user_rated_movies = self.ratings[self.ratings['userId'] == user_id]
        
        if user_rated_movies.empty:
            return pd.DataFrame() # No ratings from this user
        
        # Get genres of movies the user liked (e.g., rated >= 3.5)
        liked_movies = user_rated_movies[user_rated_movies['rating'] >= 3.5]
        if liked_movies.empty:
            # If no high ratings, consider all rated movies or provide popular movies
            return self.popular_movies(by='count', num_recommendations=num_recommendations)

        # Merge with movie genre data
        liked_movies_with_genres = pd.merge(liked_movies, self.movies_content, on='movieId', how='inner')
        
        # Create a user profile (average of genres of liked movies)
        user_profile = liked_movies_with_genres[self.genres_list].mean()

        if user_profile.empty or user_profile.isnull().all():
            return pd.DataFrame() # Could not create a user profile

        # Calculate similarity between user profile and all movies
        movie_features = self.movies_content[self.genres_list]
        
        # Handle cases where movie_features might be empty or not aligned
        if movie_features.empty:
            return pd.DataFrame()

        # Calculate dot product (cosine similarity since features are normalized implicitly if binary)
        movie_scores = movie_features.dot(user_profile)
        
        # Exclude already watched movies
        already_watched = self._get_already_watched_movies(user_id)
        unwatched_movies = self.movies_content[~self.movies_content['movieId'].isin(already_watched)]
        
        # Merge scores back to unwatched movies and sort
        unwatched_movies_with_scores = unwatched_movies.set_index('movieId').join(movie_scores.rename('score'))
        top_movie_ids = unwatched_movies_with_scores.sort_values(by='score', ascending=False).index.tolist()[:num_recommendations]

        return self.movies[self.movies['movieId'].isin(top_movie_ids)]

    def _hybrid_recommendation(self, user_id, num_recommendations):
        # Generate recommendations from each method
        user_based_recs = self._user_based_recommendation(user_id, num_recommendations * 2) # Get more to allow for overlap
        item_based_recs = self._item_based_recommendation(user_id, num_recommendations * 2)
        svd_recs = self._svd_recommendation(user_id, num_recommendations * 2)
        content_based_recs = self._content_based_recommendation(user_id, num_recommendations * 2)

        # Combine and aggregate scores
        # A simple approach: combine movie IDs and count occurrences (or average predicted scores)
        all_recs = pd.concat([user_based_recs, item_based_recs, svd_recs, content_based_recs]).drop_duplicates()
        
        # For simplicity, let's just take the top N most frequently recommended movies across methods
        # If we had actual predicted scores from each method, we would average them.
        # As a placeholder, we can assign a score to each recommended movie
        
        # A more sophisticated hybrid approach would combine the predicted scores from each model
        # For this example, we'll use a simple ranking/voting mechanism.
        
        combined_scores = {}

        # Add scores from user_based
        if not user_based_recs.empty:
            for movie_id in user_based_recs['movieId']:
                combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 1 # Simple vote

        # Add scores from item_based
        if not item_based_recs.empty:
            for movie_id in item_based_recs['movieId']:
                combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 1

        # Add scores from SVD
        if not svd_recs.empty:
            for movie_id in svd_recs['movieId']:
                combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 1

        # Add scores from content_based
        if not content_based_recs.empty:
            for movie_id in content_based_recs['movieId']:
                combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 1

        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_movie_ids = [movie_id for movie_id, score in sorted_recs[:num_recommendations]]

        return self.movies[self.movies['movieId'].isin(top_movie_ids)]

    def recommend_similar_movies(self, movie_id, num_recommendations=10):
        # Calculate content-based similarity (genre-based)
        if movie_id not in self.movies_content['movieId'].values:
            return pd.DataFrame() # Movie not found

        target_movie = self.movies_content[self.movies_content['movieId'] == movie_id]
        if target_movie.empty:
            return pd.DataFrame()

        target_features = target_movie[self.genres_list].values
        
        # Calculate cosine similarity with all other movies
        all_movie_features = self.movies_content[self.genres_list].values
        similarities = cosine_similarity(target_features, all_movie_features).flatten()
        
        # Create a series of similarities indexed by movieId
        movie_similarities = pd.Series(similarities, index=self.movies_content['movieId'])
        
        # Exclude the movie itself
        movie_similarities = movie_similarities.drop(movie_id, errors='ignore')
        
        # Get top N similar movies
        top_movie_ids = movie_similarities.sort_values(ascending=False).index.tolist()[:num_recommendations]
        
        return self.movies[self.movies['movieId'].isin(top_movie_ids)]

    def popular_movies(self, by='count', num_recommendations=10):
        if by == 'count':
            popular = movie_popularity_stats.sort_values('total_ratings', ascending=False)
        elif by == 'rating':
            popular = movie_popularity_stats.sort_values('avg_rating', ascending=False)
        else:
            raise ValueError("Invalid 'by' parameter. Choose 'count' or 'rating'.")
        
        top_movie_ids = popular['movieId'].tolist()[:num_recommendations]
        # FIXED: Merge with movie_popularity_stats to include rating counts
        result_movies = self.movies[self.movies['movieId'].isin(top_movie_ids)]
        result_with_stats = pd.merge(result_movies, movie_popularity_stats[['movieId', 'total_ratings', 'avg_rating']], 
                                     on='movieId', how='left')
        return result_with_stats

    def recommend_for_new_user(self, genre_preferences, num_recommendations=10):
        # genre_preferences is a dictionary like {'Action': 5, 'Comedy': 3, 'Drama': 4}
        # Normalize preferences to sum to 1 or scale to 0-1
        total_pref_score = sum(genre_preferences.values())
        if total_pref_score == 0:
            return self.popular_movies(by='count', num_recommendations=num_recommendations)

        normalized_preferences = {genre: score / total_pref_score for genre, score in genre_preferences.items()}
        
        # Create a pseudo-user profile vector
        user_profile_vector = pd.Series(0.0, index=self.genres_list)
        for genre, score in normalized_preferences.items():
            if genre in user_profile_vector.index:
                user_profile_vector[genre] = score
        
        # Calculate similarity with all movies based on genres
        movie_features = self.movies_content[self.genres_list]
        
        # Ensure genres are in the same order and are numerical
        if movie_features.empty:
            return pd.DataFrame()

        movie_scores = movie_features.dot(user_profile_vector)
        
        # Get top N movies based on scores
        top_movie_ids = movie_scores.sort_values(ascending=False).index.tolist()[:num_recommendations]
        return self.movies[self.movies['movieId'].isin(top_movie_ids)]


print("Recommendation algorithms defined.")

# --- 8. Model Evaluation (based on PDF Page 12) ---
# Function to evaluate RMSE, Precision, Recall for a given recommendation method

def evaluate_model(recommender_system, method, test_ratings_df, num_recommendations=10):
    rmse_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Get unique users from the test set
    test_users = test_ratings_df['userId'].unique()

    for user_id in test_users:
        # Get actual movies rated by the user in the test set
        actual_movies = test_ratings_df[test_ratings_df['userId'] == user_id]['movieId'].tolist()
        
        # Generate recommendations for the user using the specified method
        recommended_movies_df = recommender_system.recommend_for_user(user_id, method=method, num_recommendations=num_recommendations)
        recommended_movies = recommended_movies_df['movieId'].tolist()
        
        # Calculate RMSE (only if method provides ratings, like SVD)
        if method == 'svd' and recommender_system.svd_preds_df is not None:
            user_actual_ratings = test_ratings_df[test_ratings_df['userId'] == user_id].set_index('movieId')['rating']
            predicted_ratings_for_test_movies = recommender_system.svd_preds_df.loc[user_id, user_actual_ratings.index].fillna(recommender_system.ratings['rating'].mean()) # Fill any missing with global mean
            
            common_movies = list(set(user_actual_ratings.index) & set(predicted_ratings_for_test_movies.index))
            if common_movies:
                rmse = np.sqrt(mean_squared_error(user_actual_ratings[common_movies], predicted_ratings_for_test_movies[common_movies]))
                rmse_list.append(rmse)

        # Calculate Precision and Recall
        if not recommended_movies: # No recommendations made
            precision_list.append(0)
            recall_list.append(0)
            f1_list.append(0)
            continue

        true_positives = len(set(recommended_movies) & set(actual_movies))
        
        precision = true_positives / len(recommended_movies)
        recall = true_positives / len(actual_movies) if len(actual_movies) > 0 else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

    avg_rmse = np.mean(rmse_list) if rmse_list else np.nan
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    return {'method': method, 'RMSE': avg_rmse, 'Precision': avg_precision, 'Recall': avg_recall, 'F1-Score': avg_f1}

print("Model evaluation function defined.")

# --- 9. Main Execution Block / Example Usage ---

if __name__ == "__main__":
    # Split data into training and testing sets for evaluation
    # Use a stratified split to ensure similar user distribution in train/test
    # For robust evaluation, typically cross-validation or a specific user-based split is used.
    # Here, we'll use a simple random split for demonstration.
    
    # Split ratings into training and test sets (e.g., 80% train, 20% test)
    # Ensure that test set contains users with enough ratings to evaluate
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42, stratify=ratings['userId'] if 'userId' in ratings.columns else None)
    
    # Filter test_ratings to ensure users have at least one rating in the test set (and also in training for recommender)
    # This is a simplification; in real systems, train and test sets should be carefully constructed.
    
    # Initialize the recommender system with the training data
    recommender = MovieRecommendationSystem(train_ratings, movies)

    print("\n--- Example Usage ---")

    # Example 1: Get recommendations for a specific user using Hybrid method
    user_id_example = 1 # Example user ID from the dataset
    if user_id_example in ratings['userId'].unique():
        print(f"\nRecommendations for user {user_id_example} using Hybrid method:")
        hybrid_recs = recommender.recommend_for_user(user_id_example, method='hybrid', num_recommendations=5)
        print(hybrid_recs[['title', 'genres', 'year']])
    else:
        print(f"User {user_id_example} not found in dataset. Skipping user recommendations.")

    # Example 2: Find similar movies to a given movie (e.g., Toy Story - movieId 1)
    movie_id_example = 1
    if movie_id_example in movies['movieId'].unique():
        print(f"\nMovies similar to {movies[movies['movieId'] == movie_id_example]['title'].values[0]} (using content-based similarity):")
        similar_movies = recommender.recommend_similar_movies(movie_id_example, num_recommendations=5)
        print(similar_movies[['title', 'genres', 'year']])
    else:
        print(f"Movie ID {movie_id_example} not found. Skipping similar movie recommendations.")

    # Example 3: Get popular movies by rating count
    print("\nTop 5 popular movies by rating count:")
    popular_by_count = recommender.popular_movies(by='count', num_recommendations=5)
    print(popular_by_count[['title', 'genres', 'total_ratings']]) # total_ratings is from movie_popularity_stats
    

    # Example 4: Get recommendations for a new user based on genre preferences
    print("\nRecommendations for a new user with specific genre preferences:")
    new_user_genre_prefs = {
        'Action': 5,
        'Adventure': 4,
        'Sci-Fi': 5,
        'Drama': 2,
        'Comedy': 3,
        'Thriller': 4
    }
    new_user_recs = recommender.recommend_for_new_user(new_user_genre_prefs, num_recommendations=5)
    print(new_user_recs[['title', 'genres', 'year']])


    print("\n--- Model Evaluation Results ---")
    evaluation_methods = ['user_based', 'item_based', 'svd', 'content_based', 'hybrid']
    results = []

    for method in evaluation_methods:
        print(f"Evaluating {method} method...")
        # Ensure that test_ratings contains users that are also in the training set for user/item based methods
        # and that they have movies for ground truth
        eval_result = evaluate_model(recommender, method, test_ratings, num_recommendations=10)
        results.append(eval_result)

    results_df = pd.DataFrame(results).set_index('method')
    print(results_df)

    # Optional: Plotting evaluation results
    results_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
    plt.title('Recommendation System Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if not results_df['RMSE'].isnull().all():
        results_df['RMSE'].plot(kind='bar', figsize=(8, 5), color='skyblue')
        plt.title('Recommendation System RMSE')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    print("\nRecommendation system script execution complete.")