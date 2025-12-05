import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple, Any, Deque
from collections import deque
import rospy

class SimilarityMemoryCache:
    """
    A memory cache that stores key-value pairs up to a maximum size.
    Retrieval is done by finding keys with high semantic similarity.
    """

    def __init__(self,
                 threshold: float = 0.6,
                 max_size: Optional[int] = None,
                 model_name: str = 'all-MiniLM-L6-v2'):
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.max_size = max_size
        self.memory: Deque[Tuple[str, Any, np.ndarray]] = deque(maxlen=self.max_size)

        rospy.loginfo(f"Loading embedding model '{model_name}'...")
        try:
            self.model = SentenceTransformer(model_name)
            rospy.loginfo("Embedding model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load model '{model_name}': {e}")
            raise

    def _get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)
        return embedding.reshape(1, -1)

    def add(self, key: str, value: Any):
        if not isinstance(key, str) or not key:
            rospy.logwarn("Key must be a non-empty string.")
            return

        existing_item = None
        for item in self.memory:
            stored_key, _, _ = item
            if stored_key == key:
                existing_item = item
                break

        if existing_item:
            rospy.logwarn(f"Key '{key}' already in memory. Removing old entry.")
            self.memory.remove(existing_item)

        rospy.loginfo(f"Adding memory with key: '{key}'")

        embedding = self._get_embedding(key)
        self.memory.append((key, value, embedding))

    def query(self, query_text: str) -> Tuple[Optional[Any], float]:
        if not self.memory:
            return None, 0.0

        query_embedding = self._get_embedding(query_text)

        best_match_key: Optional[str] = None
        best_match_value: Any = None
        best_match_score: float = -1.0

        for key, value, stored_embedding in self.memory:
            sim_score = cosine_similarity(query_embedding, stored_embedding)[0][0]
            
            if sim_score >= self.threshold:
                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match_key = key
                    best_match_value = value

        if best_match_key is not None:
            rospy.loginfo(f"Found match via key: '{best_match_key}' (Similarity: {best_match_score:.4f})")
            return best_match_value, best_match_score
        else:
            return None, 0.0