# system imports
from typing import Type, Tuple
from collections.abc import Mapping, Sequence, Set
# project imports
from models.embeddings import EmbeddingModel, ModelType
from models.clustering import WordClusterer, ClustererType
from game.state import GameState, GameStateType
from game.recommendation import RecommendationEngine, RecommendationType
from utils.visualization import GameVisualizer, VisualizerType


GameControllerType = Type['GameController']
                          
class GameController:
    """
    This game controller class handles the interaction loop with the user.
    """
    def __init__(self: GameControllerType):
        self.embedding_model: ModelType = EmbeddingModel(model_type="conceptnet")
        self.clusterer: ClustererType = WordClusterer(self.embedding_model)
        self.recommendation_engine: RecommendationType = RecommendationEngine(self.clusterer)
        self.visualizer: VisualizerType = None  # Will be set when a game starts
        self.game_state: GameStateType = None  # Will be set when a game starts
        
    def start_new_game(self: GameControllerType):
        """
        Start a new game by getting words from the user, and returns True if all goes well.
        """
        print("\n=== NEW CONNECTIONS GAME ===")
        print("Please enter the 16 words/phrases from the Connections game in any order, separated by commas.")
        print("Example: \"springboard, vacuum, spark, summertime, wintergreen, witchcraft, fallopian, bubblegum, autumn leaves, cinnamon, launchpad, inner, menthol, unforgettable, test, catalyst\"")
        
        try:
            words_input: str = input().strip()
            words: Sequence[str] = [w.strip() for w in words_input.split(',')]
            
            # Validate word count
            if len(words) != 16:
                print(f"Error: Expected 16 words, got {len(words)}")
                return False
            
            # Initialize game state and visualizer
            self.game_state = GameState(words)
            self.visualizer = GameVisualizer(self.embedding_model)
            
            print(f"Game started with the following words: {len(words)}.")
            return True
            
        except Exception as e:
            print(f"Error starting game: {e}")
            return False
            
    def get_recommendations(self: GameControllerType):
        """
        Returns a list of recommendations for the current game state. 
        Each recommendation will be a tuple (words, score, explanation).
        """
        if not self.game_state:
            print("No active game. Start a new game first.")
            return []
            
        return self.recommendation_engine.get_recommendations(self.game_state)
        
    def process_guess(self: GameControllerType):
        """
        Process a guess from the user.
        
        Returns:
            Boolean indicating if the game should continue
        """
        if not self.game_state:
            print("No active game. Start a new game first.")
            return False
            
        try:
            # Get the guess
            print("\nEnter your guess (4 comma-separated words, in any order):")
            guess_input = input().strip()
            guess = [w.strip() for w in guess_input.split(',')]
            
            # Validate word count
            if len(guess) != 4:
                print(f"Error: Expected 4 words, got {len(guess)}")
                return True
                
            # Check if all words are on the board
            for word in guess:
                if word not in self.game_state.ungrouped:
                    print(f"Error: '{word}' is not available on the board")
                    return True
                    
            # Get result
            correct = input("Was this guess correct? (y/n): ").lower().startswith('y')
            
            if correct:
                # Get category for correct guess
                print("Hooray!")
                category = input("Enter the category/color of this group: ")
                
                # Update game state
                self.game_state.make_guess(guess, True, category)
                
                # Show progress
                progress = self.game_state.get_progress()
                print(f"\nGreat! {progress['groups_found']}/4 groups found, "
                      f"{progress['words_remaining']} words remaining.")
                
            else:
                # Check if one away
                one_away = input("Aw, shucks. Was this guess 'one word away'? (y/n): ").lower().startswith('y')
                
                # Update game state and clusterer
                self.game_state.make_guess(guess, False, one_away=one_away)
                self.clusterer.handle_wrong_guess(guess, one_away)
                
                # Show progress
                progress = self.game_state.get_progress()
                print(f"\n{progress['mistakes']}/4 mistakes made, "
                      f"{progress['mistakes_remaining']} remaining. Don't give up!")
                
                # Check if game over due to mistakes
                if progress['mistakes'] >= 4:
                    print("\nGame over! You've made 4 mistakes. Better luck next time.")
                    return False
            
            # Check if game is complete
            if self.game_state.is_complete():
                print("\nCongratulations! You've outsmarted the New York Times!")
                for color, words in self.game_state.grouped.items():
                    print(f"{color}: {', '.join(words)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error processing guess: {e}")
            return True
            

    def visualize_embeddings(self: GameControllerType):
        if not self.game_state or not self.visualizer:
            print("No active game. Start a new game first.")
            return
            
        self.visualizer.visualize_words(self.game_state.ungrouped)
        

    def display_recommendations(self: GameControllerType):
        if not self.game_state:
            print("No active game. Start a new game first.")
            return
            
        recommendations = self.get_recommendations()
        if not recommendations:
            print("No recommendations available.")
            return
            
        print("\n=== RECOMMENDED GROUPS ===")
        for i, (words, score, explanation) in enumerate(recommendations, 1):
            score_pct = int(score * 100)
            print(f"{i}. {', '.join(words)}")
            print(f"   Score: {score_pct}% - {explanation}\n")
            
    def run(self: GameControllerType):
        """Run the main game loop."""
        print("Welcome to the Connections Helper!")
        
        running = True
        game_active = False
        
        while running:
            # Display menu based on game state
            if not game_active:
                print("\n=== MENU ===")
                print("1. Start new game")
                print("0. Exit")
                
                choice = input("\nEnter your choice: ")
                
                if choice == '1':
                    game_active = self.start_new_game()
                elif choice == '0':
                    running = False
                else:
                    print("Invalid choice, please try again.")
                    
            else:
                # Game is active, show game menu
                print("\n=== GAME MENU ===")
                print("1. Get recommendations")
                print("2. Make a guess")
                print("3. Visualize word embeddings")
                print("4. Start a new game")
                print("0. Exit")
                
                choice = input("\nEnter your choice: ")
                
                if choice == '1':
                    self.display_recommendations()
                elif choice == '2':
                    # Process guess returns False when game is over
                    if not self.process_guess():
                        game_active = False
                elif choice == '3':
                    self.visualize_embeddings()
                elif choice == '4':
                    game_active = self.start_new_game()
                elif choice == '0':
                    running = False
                else:
                    print("Invalid choice, please try again.")
                    
        print("Thanks for using Connections Helper.")