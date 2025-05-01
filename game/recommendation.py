from typing import Type
from game.state import GameStateType
from models.clustering import ClustererType

RecommendationType = Type['RecommendationEngine']

class RecommendationEngine:
    """
    This class handles generating recommendations (the most likely word groupings) 
    for the player to try.
    """
    def __init__(
            self: RecommendationType, 
            clusterer: ClustererType
    ):
        self.clusterer = clusterer
        

    def get_recommendations(
            self: RecommendationType, 
            game_state: GameStateType, 
            top_n: int = 5 # no. of recommendations to return
    ): # returns list of top N (words, score, explanation) tuples
        """
        Returns the top N recommended word groups.
        """
        # No recommendations needed if game is complete
        if game_state.is_complete():
            return []
            
        # Get initial clusters from K-means
        initial_clusters = self.clusterer.calculate_kmeans(
            game_state.ungrouped,
            n_groups=max(1, len(game_state.ungrouped) // 4)
        )
        
        # Adjust scores based on previous guesses
        adjusted_clusters = self.clusterer.adjust_scores_with_feedback(
            initial_clusters, 
            game_state
        )
        
        # Process one-away guesses if available
        one_away_candidates = self._process_one_away_guesses(game_state)
        
        # Combine all candidates
        all_candidates = adjusted_clusters + one_away_candidates
        
        # Sort by score and get top N
        sorted_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:top_n]
        
        # Add explanations
        results = []
        for words, score in top_candidates:
            explanation = self._generate_explanation(words, score, game_state)
            results.append((words, score, explanation))
            
        return results
        

    def _process_one_away_guesses(
            self: RecommendationType, 
            game_state: GameStateType # Current GameState instance
    ): # returns list of (words, score) tuples derived from one-away guesses
        
        if not game_state.one_away_guesses:
            return []
            
        all_candidates = []
        
        for one_away_guess in game_state.one_away_guesses:
            # Find the most likely word to replace
            one_away_words = list(one_away_guess)
            
            # For each word in the guess
            for i in range(len(one_away_words)):
                # Create a temporary set with this word removed
                remaining_words = one_away_words.copy()
                excluded_word = remaining_words.pop(i)
                
                # Find all ungrouped words not in this guess
                potential_replacements = game_state.ungrouped - set(one_away_words)
                
                # Skip if no potential replacements
                if not potential_replacements:
                    continue
                
                # Find the word that best completes the group
                best_replacement = None
                best_score = -float('inf')
                
                for candidate in potential_replacements:
                    # Create candidate group
                    candidate_group = remaining_words + [candidate]
                    
                    # Score this group
                    score = self.clusterer.calculate_cohesion(candidate_group)
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = candidate
                
                if best_replacement:
                    # Create the improved group
                    improved_group = remaining_words + [best_replacement]
                    # Boost score for one-away derived candidates
                    all_candidates.append((improved_group, best_score * 1.5))  
        
        return all_candidates
    

    def _generate_explanation(
            self: RecommendationType, 
            words,  # List of words in the recommendation
            score,  # score of recommendation
            game_state: GameStateType # current GameState instance
    ) -> str: # returns human-readable explanation for a recommendation
        # Basic confidence level based on score
        if score > 0.6:
            confidence = "High confidence"
        elif score > 0.3:
            confidence = "Medium confidence"
        elif score > 0.1:
            confidence = "Low confidence"
        else:
            confidence = "Very low confidence"
            
        # Check if derived from one-away guess
        for one_away in game_state.one_away_guesses:
            if len(set(words).intersection(one_away)) >= 3:
                return f"{confidence} - Based on previous 'one word away' guess"
                
        # Default explanation
        return f"{confidence} - Based on semantic similarity"