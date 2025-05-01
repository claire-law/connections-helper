# system imports
import numpy as np
from typing import Type, Tuple
from collections.abc import Mapping, Sequence, Set
import pickle
import os
# project imports

ModelType = Type['EmbeddingModel']

class EmbeddingModel:
    """
    Manages word embeddings from different sources.
    
    This class provides a unified interface for accessing different types of 
    word embeddings (ConceptNet, Word2Vec, BERT) with proper fallback mechanisms
    for out-of-vocabulary words.
    """
    
    def __init__(
            self: ModelType, 
            model_type: str = "conceptnet", #  type of embedding model to use: "conceptnet", "word2vec", "bert"
            embedding_dim: int = 300, # Dimensionality of the word vectors
            models_dir: str = 'models/saved_embeddings'
    ):
        self.model_type: str = model_type
        self.embedding_dim: int = embedding_dim
        self.models_dir: str = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model = self._load_model(model_type)

        self.cache = {}  # Cache for previously computed embeddings
        
    def _load_model(
            self: ModelType,
            model_type: str # which embedding model to load: word2vec, conceptnet, bert
    ): # returns loaded embedding model object for the specified model
        
        model_path = os.path.join(self.models_dir, f"{model_type}.kv")
        
        # Check if model file exists locally
        if os.path.exists(model_path):
            print(f"Loading {model_type} embeddings from {model_path}...")
            from gensim.models import KeyedVectors
            return KeyedVectors.load(model_path)
        else:
            print('No local model. Downloading embedding model...')
            if model_type == "conceptnet":
                print(f"Loading ConceptNet embeddings...")
                import gensim.downloader as api # gensim is a popular nlp library
                model = api.load("conceptnet-numberbatch-17-06-300")
                print(f"Saving ConceptNet embeddings to {model_path}...")
                model.save(model_path)
                return model
            
            elif model_type == "word2vec":
                print(f"Loading Word2Vec embeddings...")
                import gensim.downloader as api 
                model = api.load('word2vec-google-news-300')
                print(f"Saving Word2Vec embeddings to {model_path}...")
                model.save(model_path) 
                return model
                        
            # elif model_type == "bert":
            #     print(f"Loading BERT embeddings...")
            #     from transformers import BertModel, BertTokenizer
            #     from transformers import AutoTokenizer, AutoModel
            #     model = AutoModel.from_pretrained("bert-base-uncased")
            #     # model= {'model': BertModel.from_pretrained('bert-base-uncased'),
            #             # 'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')}
            #     print(f"Saving BERT model to {model_path}...")
            #     model.save_pretrained("models/saved_embeddings/bert-base-uncased")  # HuggingFace
            #     return model
            else:
                raise ValueError(f"Unsupported embedding model type: {model_type}")
            


    def get_embedding(
            self: ModelType, 
            word: str # the word we want the embedding for
    ): # returns Numpy array with word embedding
        """
        Get embedding for a word. Handles OOV words.
        """
        # Check cache first
        if word in self.cache:
            return self.cache[word]
        
        word = word.lower().strip() # Convert word to lowercase
        
        # Handle different model types
        if self.model_type == "conceptnet":
            # Try different forms of the word with ConceptNet
            try:
                # Try with ConceptNet format
                embedding = self.model[f'/c/en/{word}']
            except KeyError:
                try:
                    # Try direct lookup
                    embedding = self.model[word]
                except KeyError:
                    # Return zeros for OOV words
                    print(f"Warning: '{word}' not found in embeddings")
                    embedding = np.zeros(self.embedding_dim)
        
        elif self.model_type == "word2vec":
            # Word2Vec lookup
            try:
                embedding = self.model[word]
            except KeyError:
                print(f"Warning: '{word}' not found in Word2Vec embeddings")
                embedding = np.zeros(self.embedding_dim)
        
        elif self.model_type == "bert":
            # TODO: for BERT, this would involve tokenization and running through the model
            embedding = np.zeros(self.embedding_dim)
            
        self.cache[word] = embedding # cache the result
        return embedding
        

    def get_embeddings(
            self: ModelType,
            words: Sequence[str] # List of words to get embeddings for
    ): # returns list of numpy arrays with word embeddings
        """
        Calls get_embedding for a list of words.
        """
        return [self.get_embedding(word) for word in words]
    

    def get_combined_embedding(
            self: ModelType,
            words #list of words to combine
    ): # returns numpy array containing the average of the embeddings for the given words
        """
        Returns a combined embedding for a group of words.
        """
        embeddings = self.get_embeddings(words)
        
        combined = np.mean(embeddings, axis=0) # get avg

        norm = np.linalg.norm(combined) # normalize
        if norm > 0:
            combined = combined / norm
            
        return combined