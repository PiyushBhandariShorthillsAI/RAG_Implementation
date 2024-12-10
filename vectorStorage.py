import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, embedding_dim: int):
        """
        Initialize the Faiss index.

        Args:
            embedding_dim (int): Dimension of the embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean distance)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to the Faiss index.

        Args:
            embeddings (np.ndarray): Numpy array of shape (num_vectors, embedding_dim).
        """
        if not self.index.is_trained:
            raise ValueError("FAISS index is not trained.")
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5):
        """
        Search the FAISS index for the nearest neighbors of the given query embeddings.

        Args:
            query_embeddings (np.ndarray): Numpy array of query embeddings of shape (num_queries, embedding_dim).
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            (distances, indices): Tuple of distances and indices of the nearest neighbors.
        """
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices
