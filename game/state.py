from typing import Type, Tuple
from collections.abc import Mapping, Sequence, Set

GameStateType = Type['GameState']


class GameState:
    """
    Tracks the current state of the game, including all words, grouped and ungrouped,
    as well as the history of guesses and their results.
    """
    
    def __init__(
            self:GameStateType, 
            words: Sequence[str]     # list of 16 user-inputted words
    ):
        # Validate input
        if len(set(words)) != len(words):
            raise ValueError("Word list contains duplicates")
            
        self.all_words: Set[str] = set(words)
        self.ungrouped: Set[str] = set(words)  # Words not yet correctly grouped
        self.grouped: Mapping[str, Set[str]] = {}  # Dict mapping color/category to set of words
        self.incorrect_guesses: Sequence[Set[Sequence[str]]] = []  # List of sets containing incorrect guesses
        self.one_away_guesses: Sequence[Set[Sequence[str]]] = []  # List of sets containing "one away" guesses
        self.num_mistakes: int = 0  # Counter for total mistakes
        
    def make_guess(
            self: GameStateType, 
            words: Sequence[str], # list of words in the guess
            correct: bool, # Whether the guess was correct 
            category: str =None, #color of correct guess
            one_away: bool = False # Whether incorrect guess was "one word away"
    ) -> bool:  # returns whether/not the game is still in progress
        
        # Validate that all words are on the board
        words_set = set(words)
        if not words_set.issubset(self.all_words):
            raise ValueError("Guess contains words not on the board")
            
        if correct:
            if category is None:
                raise ValueError("Category must be provided for correct guesses")
                
            self.grouped[category] = words_set   # add to grouped words
            self.ungrouped -= words_set    # remove from ungrouped
            
        else: # if guess is incorrect
            self.incorrect_guesses.append(words_set)
            self.num_mistakes += 1
            
            if one_away:
                self.one_away_guesses.append(words_set)
                
        return not self.is_complete()
    
    def is_complete(self: GameStateType) -> bool:  # returns True if all words have been grouped
        return len(self.ungrouped) == 0
    
    def get_progress(self: GameStateType):
        """
        Returns dictionary with progress statistics
        """
        return {
            "total_words": len(self.all_words),
            "words_remaining": len(self.ungrouped),
            "groups_found": len(self.grouped),
            "groups_remaining": 4 - len(self.grouped),
            "mistakes": self.num_mistakes,
            "mistakes_remaining": 4 - self.num_mistakes
        }
        
    def get_grouped_words(self: GameStateType) -> Set[str]:
        """
        Returns set of all words that have been correctly grouped.
        """
        all_grouped = set()
        for words in self.grouped.values():
            all_grouped.update(words)
        return all_grouped