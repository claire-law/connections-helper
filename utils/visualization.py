# system imports
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Type, Tuple
from collections.abc import Mapping, Sequence, Set
# project imports
from models.embeddings import EmbeddingModel, ModelType
from models.clustering import WordClusterer, ClustererType
from game.state import GameState, GameStateType
from game.recommendation import RecommendationEngine, RecommendationType
from utils.visualization import GameVisualizer, VisualizerType


VisualizerType = Type['GameVisualizer']

class GameVisualizer:
    """    
    This class provides visualizations, such as 2D representation of word embeddings,
    to display to user, ultimately to help them identify potential groups for their next guess.
    """
    def __init__(
            self: VisualizerType, 
            embedding_model: ModelType
    ):
        self.embedding_model = embedding_model
        

    def visualize_words(
            self: VisualizerType,
            words: Set[str] # words to visualize
    ):
        """
        Visualize word embeddings using t-SNE for dimensionality reduction.
        """
        word_list = list(words)
        
        # Get embeddings
        embeddings = [self.embedding_model.get_embedding(word) for word in word_list]
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Check if we have enough words to visualize
        if len(word_list) < 2:
            print("Need at least 2 words to visualize")
            return
            
        # Use t-SNE to reduce dimensions to 2D for visualization
        # Adjust perplexity based on number of words (shouldn't be > n-1)
        perplexity = min(5, len(word_list) - 1)
        tsne = TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            n_iter=1000
        )
        
        reduced_embeddings = tsne.fit_transform(embeddings_array) # Apply dimensionality reduction
        
        plt.figure(figsize=(10, 8))         # Create the plot
        
        # Plot each word as a point
        for i, word in enumerate(word_list):
            x, y = reduced_embeddings[i]
            plt.scatter(x, y, marker='o')
            plt.annotate(
                word,
                (x, y),
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
            )
            
        plt.title("Word Embedding Visualization", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.show()
        
        
    def visualize_group_similarities(
            self: GameVisualizer, 
            groups # List of (words, score) tuples representing potential groups
    ):
        """
        Visualize similarity between potential groups.
        """
        if len(groups) < 2:
            print("Need at least 2 groups to visualize similarities")
            return
            
        # Extract group names and scores
        group_names = [f"Group {i+1}" for i in range(len(groups))]
        scores = [score for _, score in groups]
        
        # Calculate pairwise similarities between groups
        group_embeddings = []
        for words, _ in groups:
            group_emb = self.embedding_model.get_combined_embedding(words)
            group_embeddings.append(group_emb)
            
        # Create similarity matrix
        n_groups = len(groups)
        similarity_matrix = np.zeros((n_groups, n_groups))
        
        for i in range(n_groups):
            for j in range(n_groups):
                # Reshape for sklearn cosine_similarity
                emb_i = group_embeddings[i].reshape(1, -1)
                emb_j = group_embeddings[j].reshape(1, -1)
                
                # Calculate similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix[i, j] = cosine_similarity(emb_i, emb_j)[0][0]
                
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis')
        
        # Add annotations
        for i in range(n_groups):
            for j in range(n_groups):
                text = f"{similarity_matrix[i, j]:.2f}"
                plt.text(j, i, text, ha='center', va='center', 
                         color='white' if similarity_matrix[i, j] < 0.6 else 'black')
                
        # Add labels
        plt.xticks(range(n_groups), group_names, rotation=45)
        plt.yticks(range(n_groups), group_names)
        
        # Add title and colorbar
        plt.title("Group Similarity Matrix", fontsize=16)
        plt.colorbar(label="Cosine Similarity")
        
        plt.tight_layout()
        plt.show()