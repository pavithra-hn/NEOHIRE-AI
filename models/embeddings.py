from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import pickle
import os

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the embedding model"""
        self.model_name = model_name
        self.model = None
        self.cache_dir = "embeddings_cache"
        self._load_model()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True if len(texts) > 1 else False
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """Save embeddings to cache"""
        filepath = os.path.join(self.cache_dir, f"{filename}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, filename: str) -> np.ndarray:
        """Load embeddings from cache"""
        filepath = os.path.join(self.cache_dir, f"{filename}.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()