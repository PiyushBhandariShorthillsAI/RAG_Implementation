from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from typing import List
import os

class PDFLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load_pdfs(self) -> List[str]:
        documents = []
        for file in os.listdir(self.directory):
            if file.endswith(".pdf"):
                with pdfplumber.open(os.path.join(self.directory, file)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    documents.append(text)
        print(len(documents))
        return documents


class TextSplitter:
    def __init__(self, chunk_size: int, overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap
        )

    def split_text(self, texts: List[str]) -> List[List[str]]:
        return [self.splitter.split_text(text) for text in texts]
    