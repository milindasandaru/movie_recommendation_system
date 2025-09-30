"""
Hybrid recommender system combining content-based and collaborative filtering.
"""

from .content_recommender import ContentRecommender
from .collaborative_recommender import CollaborativeRecommender


class HybridRecommender:
    """Hybrid movie recommender combining multiple approaches."""
    
    def __init__(self, content_weight=0.5, collaborative_weight=0.5):
        """Initialize the hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering recommendations
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.content_recommender = ContentRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
    
    def fit(self, movies_data, ratings_data):
        """Train both recommendation models.
        
        Args:
            movies_data: DataFrame containing movie features
            ratings_data: DataFrame containing user ratings
        """
        self.content_recommender.fit(movies_data)
        self.collaborative_recommender.fit(ratings_data)
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate hybrid recommendations for a user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movie IDs
        """
        # Get recommendations from both systems
        content_recs = self.content_recommender.recommend(user_id, n_recommendations * 2)
        collab_recs = self.collaborative_recommender.recommend(user_id, n_recommendations * 2)
        
        # Combine recommendations (placeholder implementation)
        # In practice, you would implement a proper hybrid scoring mechanism
        combined_recs = list(set(content_recs + collab_recs))
        
        return combined_recs[:n_recommendations]
