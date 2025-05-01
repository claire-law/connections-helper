from game.controller import GameController
from models.embeddings import EmbeddingModel

def main():
    game = GameController() # init a new game instance
    game.run()              # start interactive game loop

if __name__ == "__main__":
    main()