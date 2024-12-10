# import os
# import pdfplumber
# from transformers import AutoTokenizer, AutoModel
# import torch
# from typing import List
# import google.generativeai as genai
# import weaviate
# from weaviate.classes.init import Auth
# from dotenv import load_dotenv
# load_dotenv()

# class PDFLoader:
#     def __init__(self, directory: str):
#         self.directory = directory
 
#     def load_pdfs(self) -> List[str]:
#         documents = []
#         for file in os.listdir(self.directory):
#             if file.endswith(".pdf"):
#                 with pdfplumber.open(os.path.join(self.directory, file)) as pdf:
#                     text = ""
#                     for page in pdf.pages:
#                         text += page.extract_text()
#                     documents.append(text)
#         return documents
 
 
# class TextSplitter:
#     def __init__(self, chunk_size: int = 1200, overlap: int = 250):
#         self.chunk_size = chunk_size
#         self.overlap = overlap
 
#     def split_text(self, text: str) -> List[str]:
#         chunks = []
#         start = 0
#         while start < len(text):
#             end = min(start + self.chunk_size, len(text))
#             chunks.append(text[start:end])
#             start += self.chunk_size - self.overlap
#         return chunks
 
 
# class EmbeddingGenerator:
#     def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
 
#     def generate_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
#         embeddings = []
#         for text in texts:
#             inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
#         return embeddings
 
# from weaviate.auth import AuthApiKey
 
# class WeaviateStore:
#     def __init__(self, weaviate_url: str, class_name: str, api_key: str):
#         # Initialize the Weaviate client with authentication
#         self.client = weaviate.connect_to_weaviate_cloud(
#             cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
#             auth_credentials=Auth.api_key(api_key),

#         )
#         self.class_name = class_name
 
#         # Check or create the class schema
#         if not self.client.schema.exists(self.class_name):
#             schema = {
#                 "classes": [
#                     {
#                         "class": self.class_name,
#                         "vectorizer": "none",  # Custom vectorization
#                         "properties": [
#                             {
#                                 "name": "content",
#                                 "dataType": ["text"]
#                             }
#                         ]
#                     }
#                 ]
#             }
#             self.client.schema.create(schema)
 
#     def add_documents(self, documents: List[str], embeddings: List[torch.Tensor]):
#         for doc, embedding in zip(documents, embeddings):
#             self.client.data_object.create(
#                 {"content": doc},
#                 class_name=self.class_name,
#                 vector=embedding.tolist()
#             )
 
#     def query(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
#         results = self.client.query.get(
#             self.class_name, ["content"]
#         ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()
 
#         return [item["content"] for item in results["data"]["Get"][self.class_name]]
    
#     def close(self):
#         self.client.close()
 
# class RAGModel:
#     def __init__(self, directory: str, weaviate_url: str, class_name: str, wcd_api_key: str, gemini_api_key: str):
#         self.loader = PDFLoader(directory)
#         self.splitter = TextSplitter()
#         self.embedder = EmbeddingGenerator()
#         self.vector_store = WeaviateStore(weaviate_url, class_name, wcd_api_key)
#         self.api_key = gemini_api_key
 
#     def build_knowledge_base(self):
#         documents = self.loader.load_pdfs()
#         all_chunks = []
#         for doc in documents:
#             chunks = self.splitter.split_text(doc)
#             all_chunks.extend(chunks)
 
#         embeddings = self.embedder.generate_embeddings(all_chunks)
#         self.vector_store.add_documents(all_chunks, embeddings)
#         print("Knowledge base built successfully!")
 
#     def query(self, user_query: str, top_k: int = 5):
#         query_embedding = self.embedder.generate_embeddings([user_query])[0]
#         relevant_chunks = self.vector_store.query(query_embedding.tolist(), top_k)
#         context = " ".join(relevant_chunks)
#         return self._generate_response(user_query, context)
 
#     def _generate_response(self, query: str, context: str) -> str:
#         genai.configure(api_key=self.api_key)
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content("Explain how AI works")
#         print(response.text)
 
#     def close(self):
#         self.client.close()
 
# # Example Usage
# if __name__ == "__main__":
#     directory = "data"
#     weaviate_url = os.environ["weaviate_url"]
#     class_name = "PDFDocuments"
#     wcd_api_key = os.environ["wcd_api_key"]
#     gemini_api_key=os.environ["gemini_api_key"]
 
#     rag_model = RAGModel(directory, weaviate_url, class_name,wcd_api_key, gemini_api_key)
 
#     # Build the knowledge base
#     rag_model.build_knowledge_base()
 
#     # Query the system
#     query = "Give me the current used medicines."
#     response = rag_model.query(query)

#     print("Response:", response)

# 1200
# 250
# pypdf 
# legal
# faiss


from data_loader import PDFLoader, TextSplitter
from embeddingGenerator import EmbeddingGenerator
from vectorStorage import FaissIndexer
import numpy as np
from RAGModel import RAGModel

if __name__ == "__main__":
    pdf_directory = "data"
    # loader = PDFLoader(pdf_directory)
    # documents = loader.load_pdfs()

    # splitter = TextSplitter(chunk_size=1200, overlap=250)
    # chunks = splitter.split_text(documents)

    # total_chunks = sum(len(chunks) for chunks in chunks)
    # print(f"Total number of chunks: {total_chunks}")

    # print("Checking chunk types:")
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i}: {type(chunk)}")


    # embedding_generator = EmbeddingGenerator()
    # embeddings = embedding_generator.generate_embeddings(chunks)

    # print("Generated embeddings:")
    # print(embeddings)
    # print(f"Shape of embeddings: {embeddings.shape}")

    # embeddings_array = np.array(embeddings)

    # embedding_dim = 1024
    # faiss_indexer = FaissIndexer(embedding_dim)

    # faiss_indexer.add_embeddings(embeddings_array)

    # print("Embeddings successfully stored in FAISS.")

    # query_chunks = ["This is a query chunk of text."]
    # query_embeddings = embedding_generator.generate_embeddings(query_chunks)
    # query_embeddings_array = np.array(query_embeddings)

    # top_k = 25
    # # distances, indices = faiss_indexer.search(query_embeddings_array, top_k)

    # print(f"Distances:\n{distances}")
    # print(f"Indices:\n{indices}")

    chunk_size = 1200
    overlap = 250

    rag_model = RAGModel(pdf_directory, 1024, chunk_size, overlap)
    rag_model.build_knowledge_base()

    user_query = "Give me the details of supreme court of united states."
    response = rag_model.query(user_query, top_k=5)
    print(response)