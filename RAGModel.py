import numpy as np
from typing import List
from data_loader import PDFLoader, TextSplitter
from embeddingGenerator import EmbeddingGenerator
from vectorStorage import FaissIndexer

class RAGModel:
    def __init__(self, directory: str, embedding_dim: int, chunk_size: int, overlap: int):
        self.loader = PDFLoader(directory)  
        self.splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)  
        self.embedder = EmbeddingGenerator()  
        self.embedding_dim = embedding_dim
        self.vector_store = FaissIndexer(embedding_dim) 

    def build_knowledge_base(self):
        print("Loading documents...")
        documents = self.loader.load_pdfs()
        print(f"Loaded {len(documents)} documents.")

        all_chunks = []
        print("Splitting text into chunks...")
        for doc in documents:
            chunks = self.splitter.split_text(doc)
            all_chunks.extend(chunks)
        print(f"Total chunks: {len(all_chunks)}")

        print("Generating embeddings...")
        embeddings = self.embedder.generate_embeddings(all_chunks)
        print("Generated embeddings.")

        print("Storing embeddings in vector store...")
        self.vector_store.add_embeddings(embeddings)
        print("Knowledge base built successfully!")


    def query(self, user_query: str, top_k: int = 25):

        query_embedding = self.embedder.generate_embeddings([user_query])[0]

        query_embedding_array = np.array([query_embedding])
        distances, indices = self.vector_store.search(query_embedding_array, top_k)
        
        relevant_chunks = [self.all_chunks[idx] for idx in indices[0]]

        context = " ".join(relevant_chunks)

        return self._generate_response(user_query, context)

    def _generate_response(self, query: str, context: str) -> str:

        print("Query:", query)
        print("Context:", context)
        return f"Response to: '{query}'\nBased on context: '{context}'"
