# Movie Recommendation System

This project involves building a comprehensive movie recommendation system using the MovieLens dataset, incorporating various collaborative filtering and content-based approaches. The system includes both backend recommendation algorithms and an interactive data visualization dashboard for comprehensive analysis.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation and Setup](#installation-and-setup)
4. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
5. [Recommendation Algorithms Implemented](#recommendation-algorithms-implemented)
6. [Interactive Dashboard](#interactive-dashboard)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview

This project focuses on developing a comprehensive movie recommendation system that helps users discover content relevant to their interests from a vast array of options. The system consists of two main components:

### Backend Recommendation Engine
Implements and evaluates several recommendation algorithms:
* **Collaborative Filtering (User-Based and Item-Based):** Recommends items based on user-item interactions
* **Matrix Factorization (SVD):** Identifies latent factors from user-item interactions
* **Content-Based Filtering:** Recommends items based on their attributes and user preferences
* **Hybrid Recommendation System:** Combines the strengths of multiple approaches

### Interactive Visualization Dashboard
A React-based dashboard that provides:
* **Dataset Overview:** Comprehensive analysis of genre distribution, rating patterns, and temporal trends
* **Algorithm Performance:** Visual comparison of recommendation algorithms using RMSE, Precision, Recall, and F1-Score
* **User Preference Analysis:** Interactive heatmaps showing user-genre preferences and activity patterns
* **Movie Similarity Network:** D3.js-powered network graphs showing movie relationships

## Dataset

The project utilizes the **MovieLens latest-small dataset**, a widely-used benchmark dataset for recommendation systems research.

### Dataset Structure
* **movies.csv:** Contains movie IDs, titles, and genres
* **ratings.csv:** Contains user ratings for movies (UserID, MovieID, Rating, Timestamp)
* **links.csv:** Provides MovieID, IMDB ID, and TMDB ID mappings
* **tags.csv:** Contains user-generated tags for movies

### Dataset Statistics (as visualized in dashboard)
* **9,742 movies** across 20 genres
* **100,836 ratings** from 610 users
* **Genre Distribution:** Drama (44.8%) and Comedy (38.6%) dominate
* **Average Rating:** 3.5/5.0 with positive rating bias

## Installation and Setup

### Backend Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd movie-recommendation-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

### Dashboard Setup
The interactive dashboard is built using React with the following key libraries:
* **React:** Frontend framework with hooks (useState, useEffect, useRef)
* **Recharts:** For responsive charts (BarChart, LineChart, PieChart, ScatterChart, AreaChart)
* **D3.js:** For advanced visualizations (heatmaps, network graphs)
* **Tailwind CSS:** For styling and responsive design

#### Dashboard Features
* **Responsive Design:** Works across desktop and mobile devices
* **Interactive Charts:** Hover tooltips, drag-and-drop network nodes
* **Multiple Visualization Types:** 
  - Bar charts for genre distribution
  - Area charts for rating distribution
  - Line charts for temporal trends
  - Scatter plots for algorithm comparison
  - Heatmaps for user preferences
  - Network graphs for movie similarities
* **Modern UI:** Gradient backgrounds, smooth transitions, and contemporary design

## Data Preprocessing and Feature Engineering

### Data Cleaning
* **Missing Value Handling:** Addressed missing values (e.g., tmdbId filled with -1)
* **Duplicate Removal:** Removed duplicate entries from movies and ratings datasets
* **Data Integrity Checks:** Ensured consistent movie IDs and valid rating ranges (0.5-5.0)

### Feature Engineering
* **Movie Features:**
  - Year extraction from movie titles
  - One-hot encoded genre features
  - Movie popularity metrics (total_ratings, avg_rating, rating_std)
  
* **User Features:**
  - User activity metrics (total_ratings, avg_rating, rating_std)
  - Rating behavior patterns
  
* **Temporal Features:**
  - Time-based features (year, month, day, day of week) from ratings timestamps
  
* **Data Transformations:**
  - Normalized ratings by user mean
  - Min-Max scaling for dense features
  - User-item matrix creation for collaborative filtering

### Outlier Handling
* Removed users with extreme rating patterns
* Handled cold-start problems for new users/movies
* Applied statistical outlier detection methods

## Recommendation Algorithms Implemented

### 1. User-Based Collaborative Filtering
Identifies users with similar tastes and recommends movies liked by similar users.
- **Dashboard Insight:** Shows user clustering patterns in heatmap visualization

### 2. Item-Based Collaborative Filtering  
Recommends movies similar to those the user has already liked.
- **Performance:** Good precision (0.82) as shown in dashboard

### 3. Matrix Factorization (SVD)
Uses Singular Value Decomposition to uncover latent factors in user-item interactions.
- **Performance:** Lowest RMSE (0.85) according to dashboard metrics

### 4. Content-Based Filtering
Builds user profiles based on movie genres and attributes.
- **Performance:** Effective for cold-start problems (F1: 0.70)

### 5. Hybrid Recommendation System
Combines multiple approaches with weighted scoring.
- **Performance:** Best overall performance (F1: 0.82, RMSE: 0.82)

## Interactive Dashboard

### Navigation Tabs
1. **Dataset Overview:** Genre distribution, rating patterns, temporal trends
2. **Algorithm Performance:** RMSE, Precision, Recall, F1-Score comparisons
3. **User Preferences:** Interactive heatmaps and activity patterns
4. **Movie Similarity:** Network graphs and genre distributions

### Key Visualizations

#### D3.js Custom Visualizations
* **User-Genre Preference Heatmap:** 
  - Interactive color-coded matrix showing user preferences
  - Hover tooltips with detailed information
  - Helps identify user clustering opportunities

* **Movie Similarity Network Graph:**
  - Force-directed layout with draggable nodes
  - Node size represents movie rating
  - Edge thickness shows similarity strength
  - Color coding by movie genre

#### Recharts Visualizations
* **Responsive Charts:** Automatically adjust to screen size
* **Custom Tooltips:** Enhanced hover information
* **Multiple Chart Types:** Bar, Line, Area, Scatter, Pie charts
* **Professional Styling:** Modern color schemes and animations

### Dashboard Insights
The dashboard provides actionable insights:
* **Data Characteristics:** Comprehensive dataset statistics
* **Algorithm Performance:** Comparative analysis of different approaches
* **User Behavior:** Patterns in rating behavior and preferences
* **Content Analysis:** Genre popularity and temporal trends

## Model Evaluation

### Evaluation Metrics
* **RMSE (Root Mean Squared Error):** Measures rating prediction accuracy
* **Precision:** Proportion of recommended items that are relevant
* **Recall:** Proportion of relevant items that are recommended
* **F1-Score:** Harmonic mean of Precision and Recall

### Evaluation Process
1. Split user ratings into training and test sets
2. Generate recommendations using each algorithm
3. Calculate metrics by comparing predictions against test set
4. Visualize results in interactive dashboard

### Performance Results (from Dashboard)
| Algorithm | RMSE | Precision | Recall | F1-Score |
|-----------|------|-----------|--------|----------|
| User-Based CF | 0.94 | 0.78 | 0.65 | 0.71 |
| Item-Based CF | 0.89 | 0.82 | 0.69 | 0.75 |
| SVD Matrix | 0.85 | 0.85 | 0.72 | 0.78 |
| Content-Based | 0.91 | 0.79 | 0.63 | 0.70 |
| **Hybrid System** | **0.82** | **0.88** | **0.76** | **0.82** |

## Usage

### Backend Usage
```python
import pandas as pd

# Load dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Preprocessing
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['title'] = movies['title'].str.replace(r' \(\d{4}\)$', '', regex=True)
genres = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)

# Initialize recommender system
recommender = MovieRecommendationSystem(ratings, movies)

# Get recommendations
user_id = 1
recommendations = recommender.recommend_for_user(user_id, method='hybrid')
print(f"Recommendations for user {user_id}:")
print(recommendations)

# Find similar movies
movie_id = 1
similar_movies = recommender.recommend_similar_movies(movie_id)
print(f"Movies similar to {movies[movies['movieId'] == movie_id]['title'].values[0]}:")
print(similar_movies)

# Get popular movies
popular_by_rating = recommender.popular_movies(by='rating')
print("Popular movies by rating:")
print(popular_by_rating)

# New user recommendations
genre_preferences = {
    'Action': 5,
    'Adventure': 4,
    'Sci-Fi': 5,
    'Drama': 2,
    'Comedy': 3
}
new_user_recs = recommender.recommend_for_new_user(genre_preferences)
print("Recommendations for new user:")
print(new_user_recs)
```

### Dashboard Usage
The React dashboard provides an intuitive interface for exploring the recommendation system:

1. **Launch the dashboard** (assuming it's integrated into your React application)
2. **Navigate between tabs** to explore different aspects of the system
3. **Interact with visualizations:**
   - Hover over charts for detailed information
   - Drag nodes in the network graph
   - Click on different algorithm comparisons
4. **Analyze insights** from the comprehensive data views

## Future Improvements

### Algorithm Enhancements
* **Deep Learning Models:** Implement neural collaborative filtering
* **Real-time Learning:** Add online learning capabilities
* **Context-aware Recommendations:** Include temporal and contextual factors
* **Cold-start Solutions:** Improve handling of new users and items

### Dashboard Enhancements
* **Real-time Data Updates:** Live dashboard updates with new ratings
* **User Interaction Tracking:** Analytics on dashboard usage
* **Export Capabilities:** Download charts and reports
* **Mobile Optimization:** Enhanced mobile user experience
* **Advanced Filtering:** More sophisticated data filtering options

### Scalability Improvements
* **Distributed Computing:** Implement Spark for large-scale processing
* **Caching Strategies:** Optimize recommendation generation speed
* **API Integration:** RESTful API for recommendation services
* **A/B Testing Framework:** Compare different recommendation strategies

## Contributing

We welcome contributions to improve both the recommendation algorithms and the visualization dashboard. Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement your changes** with appropriate tests
4. **Update documentation** including README and code comments
5. **Submit a pull request** with a clear description of changes

### Areas for Contribution
* **Algorithm Development:** New recommendation techniques
* **Dashboard Features:** Additional visualizations and interactions
* **Performance Optimization:** Speed and memory improvements
* **Documentation:** Improved guides and examples

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

* **MovieLens Dataset:** University of Minnesota GroupLens Research Project
* **Visualization Libraries:** Recharts, D3.js, and the open-source community
* **React Ecosystem:** For providing excellent tools for building interactive UIs
