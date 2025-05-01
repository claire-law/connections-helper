#system imports
from typing import Type, Tuple
from collections.abc import Mapping, Sequence, Set
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
# project imports
from models.embeddings import EmbeddingModel, ModelType

ClustererType = Type['WordClusterer']

class WordClusterer:
    """
    Clusters words based on embeddings and scores potential groups.
    
    This class implements various clustering algorithms to identify
    potential word groups based on their semantic relationships.
    It also provides methods for scoring and ranking these groups.
    """
    
    def __init__(
            self: ClustererType, 
            embedding_model: ModelType
    ):
        self.embedding_model = embedding_model
        self.negative_patterns = []  # Semantic patterns to avoid (from wrong guesses)
        

    def calculate_kmeans(
            self: ClustererType, 
            words, # set of words to cluster
            n_groups: int = 4, # no. of groups to create must be 4, unless testing otherwise
            words_per_group: int = 4 # no. of words per group must be 4, unless testing otherwise
    ): #  Returns list of (group, score) tuples, sorted by score
        """
        Calculate K-means clustering on word embeddings.
        This is the main method for finding potential word groups.
        """
        # Validate input
        total_words = len(words)
        if total_words < n_groups:
            n_groups = total_words  # Adjust if we have fewer words than groups
            
        # Convert words to list for indexing
        words_list = list(words)
        
        # Get embeddings for all words
        embeddings_list = []
        for word in words_list:
            embeddings_list.append(self.embedding_model.get_embedding(word))
        
        # Convert to numpy array for KMeans
        embeddings_array = np.array(embeddings_list)
        
        # Apply KMeans with multiple initializations for better results
        kmeans = KMeans(
            n_clusters=n_groups, 
            random_state=0, 
            n_init=10  # Try 10 different initializations
        ).fit(embeddings_array)
        
        # Extract cluster labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # Group words by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(words_list[i])
        
        # We may need to balance clusters to ensure equal sizes
        if words_per_group is not None:
            clusters = self._balance_clusters(
                clusters, 
                words_list, 
                labels, 
                centers, 
                embeddings_array, 
                words_per_group
            )
        
        # Calculate scores for each cluster
        cluster_scores = []
        for cluster_words in clusters.values():
            score = self.calculate_cohesion(cluster_words)
            cluster_scores.append((cluster_words, score))
        
        # Sort by score and return
        return sorted(cluster_scores, key=lambda x: x[1], reverse=True)
    

    def _balance_clusters(
            self: ClustererType, 
            clusters,       # Dictionary mapping cluster labels to lists of words
            words_list,     # Original list of words
            labels,         # cluster labels for each word
            centers,        # cluster centers
            embeddings,     # word embeddings
            target_size     # target number of words per cluster
    ): #   returns dictionary of balanced clusters to ensure they have equal sizes.

        # Count words per cluster
        cluster_counts = {label: len(words) for label, words in clusters.items()}
        
        # Find clusters that need words and those with excess
        deficit_clusters = [c for c, count in cluster_counts.items() if count < target_size]
        excess_clusters = [c for c, count in cluster_counts.items() if count > target_size]
        
        # Create a copy of the original clusters to modify
        balanced_clusters = {label: list(words) for label, words in clusters.items()}
        
        # Balance clusters by moving words from excess to deficit clusters
        while deficit_clusters and excess_clusters:
            deficit = deficit_clusters[0]
            excess = excess_clusters[0]
            
            # Find word in excess cluster that's closest to deficit cluster center
            excess_indices = [i for i, label in enumerate(labels) if label == excess]
            deficit_center = centers[deficit]
            
            # Compute distances to deficit center
            distances = []
            for i in excess_indices:
                dist = np.linalg.norm(embeddings[i] - deficit_center)
                distances.append((i, dist))
            
            # Sort by distance (ascending)
            distances.sort(key=lambda x: x[1])
            
            # Move the closest word
            closest_idx, _ = distances[0]
            word_to_move = words_list[closest_idx]
            
            # Update balanced clusters
            balanced_clusters[excess].remove(word_to_move)
            balanced_clusters[deficit].append(word_to_move)
            
            # Update counts
            cluster_counts[excess] -= 1
            cluster_counts[deficit] += 1
            
            # Update deficit/excess lists
            if cluster_counts[deficit] == target_size:
                deficit_clusters.pop(0)
            if cluster_counts[excess] == target_size:
                excess_clusters.pop(0)
        
        return balanced_clusters
        
        
    def calculate_cohesion(
            self: ClustererType, 
            words # list of words in the group
    ): # returns cohesion score (the higher the better)
        """
        Calculate cohesion score for a group of words based on pairwise similarity.
        """
        if len(words) <= 1:
            return 0
        
        # Get embeddings for all words
        embeddings = [self.embedding_model.get_embedding(word) for word in words]
        
        # Calculate pairwise cosine similarities
        total_similarity = 0
        num_pairs = 0
        
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                # Get embeddings
                emb_i = embeddings[i].reshape(1, -1)  # Reshape for sklearn
                emb_j = embeddings[j].reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(emb_i, emb_j)[0][0]
                
                total_similarity += similarity
                num_pairs += 1
        
        # Average similarity (cohesion score)
        return total_similarity / num_pairs if num_pairs > 0 else 0
    

    def handle_wrong_guess(
            self: ClustererType, 
            words, # list of words in the wrong guess
            one_away:bool = False # whether/not the guess was one word away from being correct
    ):
        """
        Learn from wrong guesses to improve future recommendations.
        """
        # Extract semantic features from the wrong guess
        guess_embedding = self.embedding_model.get_combined_embedding(words)
        
        # Add to negative patterns with lower weight if it was "one away"
        weight = 0.5 if one_away else 1.0
        self.negative_patterns.append((guess_embedding, weight))
        

    def adjust_scores_with_feedback(
            self: ClustererType, 
            candidate_groups, # list of (words, score) tuples to adjust
            game_state        # current game state with wrong guesses history
    ): # Returns adjusted list of (words, score) tuples
        """
        Adjust group scores based on previous wrong guesses.
        """
        adjusted_groups = []
        
        for words, score in candidate_groups:
            words_set = set(words)
            adjustment_factor = 1.0
            
            # Check for exact matches or overlap with previous wrong guesses
            for wrong_guess in game_state.incorrect_guesses:
                # Exact match - severe penalty
                if words_set == wrong_guess:
                    adjustment_factor *= 0.1
                    break
                    
                # Significant overlap (3 words) - major penalty
                overlap = len(words_set.intersection(wrong_guess))
                if overlap >= 3:
                    adjustment_factor *= 0.3
                # Moderate overlap (2 words) - minor penalty    
                elif overlap == 2:
                    adjustment_factor *= 0.8
            
            # Apply semantic similarity penalties from negative patterns
            group_embedding = self.embedding_model.get_combined_embedding(words)
            for pattern, weight in self.negative_patterns:
                similarity = cosine_similarity(
                    group_embedding.reshape(1, -1), 
                    pattern.reshape(1, -1)
                )[0][0]
                
                # Apply penalty proportional to similarity
                if similarity > 0.7:  # Only penalize high similarity
                    adjustment_factor *= (1 - ((similarity - 0.7) * weight))
            
            # Add adjusted score
            adjusted_groups.append((words, score * adjustment_factor))
        
        # Sort by adjusted score
        return sorted(adjusted_groups, key=lambda x: x[1], reverse=True)