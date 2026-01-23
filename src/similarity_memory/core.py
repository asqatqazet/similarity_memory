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
                 threshold: float = 0.3,
                 max_size: Optional[int] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 fact_file: str = 'fact.json'):

        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.max_size = max_size
        self.memory: Deque[Tuple[str, Any, np.ndarray]] = deque(maxlen=self.max_size)

        # 1. Load the embedding model
        rospy.loginfo(f"Loading embedding model '{model_name}'...")
        try:
            self.model = SentenceTransformer(model_name)
            rospy.loginfo("Embedding model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load model '{model_name}': {e}")
            raise

        # 2. Load default memories from file
        self.load_memories_from_json(fact_file)

        rospy.loginfo("Initialization complete.")

    def load_memories_from_json(self, file_path: str):
        """
        Parses the fact.json file and populates the memory cache.
        """
        if not os.path.exists(file_path):
            rospy.logwarn(f"Fact file '{file_path}' not found. Memory remains empty.")
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Access the "memory" list from the JSON structure
            facts_to_load = data.get("memory", [])

            rospy.loginfo(f"Populating cache with {len(facts_to_load)} memories from {file_path}...")

            for entry in facts_to_load:
                item = entry.get("item", "")
                fun_fact = entry.get("fun_facts", "")

                # Only add if the 'item' is not empty
                if item:
                    self.add(item, fun_fact)

            rospy.loginfo(f"Successfully loaded {len(self.memory)} items into memory.")

        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to parse JSON in {file_path}: {e}")
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred while loading {file_path}: {e}")

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
        best_match_score: float = float('-inf')

        for key, value, stored_embedding in self.memory:
            sim_score = cosine_similarity(query_embedding, stored_embedding)[0][0]
            if sim_score > best_match_score:
                best_match_score = sim_score
                best_match_key = key
                best_match_value = value

        if best_match_key is None:
            return None, 0.0

        if best_match_score >= self.threshold:
            rospy.loginfo(f"Found match via key: '{best_match_key}' (Similarity: {best_match_score:.4f})")
            return best_match_value, best_match_score
        else:
            rospy.loginfo(
                f"No match reached threshold {self.threshold:.2f}. Best candidate was '{best_match_key}' with similarity {best_match_score:.4f}."
            )
            return None, best_match_score


# --- MAIN BLOCK ---
if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('memory_cache_test_node', anonymous=True)

    try:
        # Create instance (assumes fact.json is in the same folder)
        cache = SimilarityMemoryCache(threshold=0.5, fact_file='fact.json')

        # Test Query
        test_query = "Tell me about the chocolate on the shelf"
        rospy.loginfo(f"Querying: '{test_query}'")

        result, score = cache.query(test_query)

        if result:
            print(f"\n[SUCCESS] Result: {result}")
            print(f"[SCORE] {score:.4f}")
        else:
            print("\n[FAILURE] No relevant memory found above threshold.")

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node execution failed: {e}")