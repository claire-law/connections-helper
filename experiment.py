# system imports
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.colors as mcolors

# project imports
from models.embeddings import EmbeddingModel
from models.clustering import WordClusterer
from game.state import GameState
from game.recommendation import RecommendationEngine


def load_dataset():
    data_dir = "data"
    data_file = os.path.join(data_dir, "huggingface.json")
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(data_file):
        print(f"Loading dataset from {data_file}")
        df = pd.read_json(data_file)
    else:
        print(f"Downloading dataset from Huggingface")
        try:
            # This requires user to be logged in via `huggingface-cli login`
            df = pd.read_json("hf://datasets/tm21cy/NYT-Connections/ConnectionsFinalDataset.json")
            df.to_json(data_file)             # Save locally for future use
            print(f"Dataset saved to {data_file}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please log in using `huggingface-cli login` and try again.")
            return None
    return df

def preprocess_games(df):
    games = []
    
    for _, row in df.iterrows():
        game_data = {
            'date': row.get('date', 'Unknown'),
            'words': row.get('words', []),
            'groups': {}
        }
        
        # extract groups from the 'answers' list of dictionaries
        if isinstance(row.get('answers'), list):
            for i, answer_dict in enumerate(row.get('answers')):
                # Use the index as color if no color is provided
                color = answer_dict.get('color', f"group_{i}")
                words = answer_dict.get('words', [])
                
                nyt_colors = ['yellow', 'green', 'blue', 'purple']
                if i < len(nyt_colors) and 'color' not in answer_dict:
                    color = nyt_colors[i]
                    
                game_data['groups'][color] = words
        
        if game_data['words'] and game_data['groups']:
            games.append(game_data)

    return games


def play_game(game_data, embedding_type):
    """
    Simulate playing a game with the given embedding model.
    Returns results including whether game was won and color-specific statistics.
    """
    # initialize models and game state
    embedding_model = EmbeddingModel(model_type=embedding_type)
    clusterer = WordClusterer(embedding_model)
    recommendation_engine = RecommendationEngine(clusterer)
    game_state = GameState(game_data['words'])
    
    color_results = {color: False for color in game_data['groups'].keys()}
    
    # Play until game is complete or 4 mistakes are made
    while not game_state.is_complete() and game_state.num_mistakes < 4:
        recommendations = recommendation_engine.get_recommendations(game_state)
        
        if not recommendations:
            break  # No more recommendations available
        
        top_words, _, _ = recommendations[0]
        
        correct = False
        matching_color = None
        
        for color, group_words in game_data['groups'].items():
            if set(top_words) == set(group_words):
                correct = True
                matching_color = color
                # Mark this color as correctly guessed
                color_results[color] = True
                break
        
        if correct:
            game_state.make_guess(top_words, True, matching_color)
        else:
            one_away = False
            for color, group_words in game_data['groups'].items():
                if len(set(top_words).intersection(set(group_words))) == 3:
                    one_away = True
                    break
            
            game_state.make_guess(top_words, False, one_away=one_away)
            clusterer.handle_wrong_guess(top_words, one_away)
    
    won = game_state.is_complete()
    
    return {
        'won': won,
        'mistakes': game_state.num_mistakes,
        'color_results': color_results
    }


def run_experiments(games, # list of game data
                    sample_size=None,  #Number of games to sample (None for all)
                    show_progress=True): #show progress bar or not
    """
    Run experiments on both ConceptNet and Word2Vec models, then
    return dictionary with experiment results.
    """
    # Sample games if requested
    if sample_size is not None and sample_size < len(games):
        # Use fixed seed for reproducibility
        np.random.seed(42)
        sampled_indices = np.random.choice(len(games), sample_size, replace=False)
        sampled_games = [games[i] for i in sampled_indices]
    else:
        sampled_games = games
    
    results = {
        'conceptnet': {
            'wins': 0,
            'games_played': 0,
            'color_wins': defaultdict(int),
            'color_total': defaultdict(int),
            'game_outcomes': []
        },
        'word2vec': {
            'wins': 0,
            'games_played': 0,
            'color_wins': defaultdict(int),
            'color_total': defaultdict(int),
            'game_outcomes': []
        }
    }
    
    # Define model types to test
    model_types = ['conceptnet', 'word2vec']
    
    for model_type in model_types:
        print(f"\nRunning experiments with {model_type} model...")
        
        # Create progress bar if requested
        game_iterator = tqdm(sampled_games) if show_progress else sampled_games
        
        for game in game_iterator:
            # Play the game
            game_result = play_game(game, model_type)
            
            # Update results
            results[model_type]['games_played'] += 1
            if game_result['won']:
                results[model_type]['wins'] += 1
            
            results[model_type]['game_outcomes'].append(game_result)
            
            # Update color-specific stats
            for color, was_correct in game_result['color_results'].items():
                results[model_type]['color_total'][color] += 1
                if was_correct:
                    results[model_type]['color_wins'][color] += 1
    
    return results


def visualize_results(results):
    os.makedirs("results", exist_ok=True)
    
    # 1. Overall win rates
    win_rates = {
        model: results[model]['wins'] / results[model]['games_played'] * 100
        for model in results
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(win_rates.keys(), win_rates.values())
    
    # percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom'
        )
    
    plt.title('Win Rate by Embedding Model')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, max(win_rates.values()) * 1.2)  # Add some space for the text
    plt.savefig('results/win_rates.png')
    plt.close()
    
    # 2. Color-specific success rates for both models
    plt.figure(figsize=(12, 8))
    
    # Get all colors across both models
    all_colors = set()
    for model in results:
        all_colors.update(results[model]['color_total'].keys())
    
    # Prepare data
    models = list(results.keys())
    x = np.arange(len(all_colors))
    width = 0.35
    
    # Sort colors by difficulty (standard NYT order: yellow, green, blue, purple)
    color_order = ['yellow', 'green', 'blue', 'purple']
    sorted_colors = sorted(all_colors, key=lambda c: color_order.index(c) if c in color_order else 999)
    
    for i, model in enumerate(models):
        color_rates = []
        for color in sorted_colors:
            if results[model]['color_total'].get(color, 0) > 0:
                rate = results[model]['color_wins'].get(color, 0) / results[model]['color_total'][color] * 100
            else:
                rate = 0
            color_rates.append(rate)
        
        offset = width * (i - 0.5)
        bars = plt.bar(x + offset, color_rates, width, label=model)
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.title('Success Rate by Color and Embedding Model')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Color Category')
    plt.xticks(x, sorted_colors)
    plt.legend()
    plt.ylim(0, 100 * 1.1)  # Add some space for the text
    plt.savefig('results/color_success_rates.png')
    plt.close()
    
    # 3. Game outcome distribution
    outcomes = {
        'conceptnet': Counter(),
        'word2vec': Counter()
    }
    
    # Simulate games again to collect outcome distribution
    for model_type in results:
        # Count outcomes (won or number of mistakes)
        for game in tqdm(results[model_type].get('game_outcomes', [])):
            if game['won']:
                outcomes[model_type]['Won'] += 1
            else:
                outcomes[model_type][f"{game['mistakes']} mistakes"] += 1
    
    # Create figure with subplots for each model
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    for i, model in enumerate(outcomes):
        # Get labels and counts
        labels = list(outcomes[model].keys())
        counts = list(outcomes[model].values())
        
        # Create pie chart
        axs[i].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        axs[i].set_title(f'{model.capitalize()} Outcomes')
        axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.savefig('results/game_outcomes.png')
    plt.close()
    
    # 4. Print summary statistics
    print("\n=== EXPERIMENT RESULTS ===")
    for model in results:
        print(f"\n{model.upper()} MODEL:")
        print(f"Games played: {results[model]['games_played']}")
        print(f"Games won: {results[model]['wins']}")
        print(f"Win rate: {win_rates[model]:.2f}%")
        
        print("\nColor-specific success rates:")
        for color in sorted_colors:
            if results[model]['color_total'].get(color, 0) > 0:
                rate = results[model]['color_wins'].get(color, 0) / results[model]['color_total'][color] * 100
                print(f"  {color}: {rate:.2f}% ({results[model]['color_wins'].get(color, 0)}/{results[model]['color_total'][color]})")


def main():
    print("Loading dataset...")
    df = load_dataset()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded with {len(df)} entries.")
    
    print("Preprocessing games...")
    games = preprocess_games(df)
    print(f"Preprocessed {len(games)} games.")
    
    sample_size = 50     # Set sample_size to None to use all games
    print(f"Running experiments on {sample_size if sample_size else 'all'} games...")
    results = run_experiments(games, sample_size=sample_size)
    
    print("Visualizing results...")
    visualize_results(results)
    
    print("Experiments completed. Visuals saved to 'results' directory.")


if __name__ == "__main__":
    main()