import faiss
import numpy as np
import logging
import time
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class FaissRetrievalIndex:
    """
    Industrial Vector Search infrastructure using FAISS.
    Handles exact and approximate nearest neighbor (ANN) retrieval.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        
        # FAISS only accepts int64 IDs. We must map string IDs to int64.
        self.int_to_str_id: Dict[int, str] = {}
        self.str_to_int_id: Dict[str, int] = {}
        self.is_trained = False

    def build_exact_index(self, item_embeddings: np.ndarray, item_ids: List[str]) -> None:
        """
        Builds an Exact Search (Brute Force) Index. 
        Use this if catalog size < 1 Million items. Guaranteed 100% recall.
        """
        logger.info(f"Building Exact FAISS Index for {len(item_ids)} items...")
        
        # We use IndexFlatIP (Inner Product) because our embeddings are L2-Normalized.
        # Normalized Inner Product == Cosine Similarity!
        base_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Wrap it to store IDs
        self.index = faiss.IndexIDMap(base_index)
        self._add_items_to_index(item_embeddings, item_ids)

    def build_ann_index(self, item_embeddings: np.ndarray, item_ids: List[str], 
                        nlist: int = 100) -> None:
        """
        Builds an Approximate Nearest Neighbor (ANN) Index using IVF.
        Use this for > 1 Million items. Trades slight accuracy drop for extreme speed.
        """
        logger.info(f"Building ANN (IVF) FAISS Index with nlist={nlist}...")
        
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # IVF (Inverted File) partitions the vector space into `nlist` clusters.
        base_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # IVF requires training to find the cluster centroids (Voronoi cells)
        logger.info("Training IVF centroids...")
        base_index.train(item_embeddings)
        self.is_trained = True
        
        self.index = faiss.IndexIDMap(base_index)
        self._add_items_to_index(item_embeddings, item_ids)

    def _add_items_to_index(self, item_embeddings: np.ndarray, item_ids: List[str]) -> None:
        """Helper function to map IDs and add embeddings to the index."""
        # Ensure float32 (FAISS requirement)
        item_embeddings = item_embeddings.astype(np.float32)
        
        # Create int64 IDs
        int_ids = np.arange(len(item_ids), dtype=np.int64)
        for integer_id, string_id in zip(int_ids, item_ids):
            self.int_to_str_id[integer_id] = string_id
            self.str_to_int_id[string_id] = integer_id
            
        self.index.add_with_ids(item_embeddings, int_ids)
        logger.info(f"Successfully added {self.index.ntotal} items to FAISS.")

    def search(self, user_embedding: np.ndarray, k: int = 100, nprobe: int = 10) -> Tuple[List[str], List[float]]:
        """
        Retrieves the Top-K items for a given user embedding.
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized.")
            
        # Ensure shape is (1, dim) and type is float32
        user_embedding = user_embedding.reshape(1, -1).astype(np.float32)
        
        # If using IVF, nprobe dictates how many neighboring clusters to search.
        # Higher nprobe = higher accuracy, slower speed.
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        start_time = time.time()
        
        # D: Distances (Scores), I: Indices (int64 IDs)
        scores, indices = self.index.search(user_embedding, k)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"FAISS Search Latency: {latency_ms:.2f} ms")

        # Map back to string IDs
        top_k_ids = [self.int_to_str_id[idx] for idx in indices[0] if idx != -1]
        top_k_scores = [float(score) for score in scores[0]]
        return top_k_ids, top_k_scores

    def save_index(self, index_path: str, mapping_path: str) -> None:
        """Saves the FAISS index and the ID mappings to disk."""
        import pickle
        faiss.write_index(self.index, index_path)
        with open(mapping_path, 'wb') as f:
            pickle.dump({'int_to_str': self.int_to_str_id, 'str_to_int': self.str_to_int_id}, f)
        logger.info("FAISS index and mappings saved successfully.")

    def load_index(self, index_path: str, mapping_path: str) -> None:
        """Loads the FAISS index and ID mappings from disk."""
        import pickle
        self.index = faiss.read_index(index_path)
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            self.int_to_str_id = mappings['int_to_str']
            self.str_to_int_id = mappings['str_to_int']
        logger.info("FAISS index and mappings loaded successfully.")