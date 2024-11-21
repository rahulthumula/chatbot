# ingest.py
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List
import os

def load_documents(directory: str) -> List:
    """
    Load documents from multiple file types in the specified directory
    """
    # Create loaders for different file types
    loaders = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader
    }
    
    documents = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            # Skip if file type not supported
            if file_extension not in loaders:
                print(f"Skipping {file} - unsupported file type")
                continue
                
            try:
                loader = loaders[file_extension](file_path)
                documents.extend(loader.load())
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return documents

def process_documents():
    """
    Load, chunk, and create vectorstore from documents
    """
    # Check if documents directory exists
    if not os.path.exists("data/documents"):
        raise Exception("data/documents directory not found. Please create it and add your documents.")
    
    # Load documents
    print("Loading documents...")
    documents = load_documents("data/documents")
    
    if not documents:
        raise Exception("No documents found in data/documents directory")
    
    print(f"\nLoaded {len(documents)} documents")
    
    # Create text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Print first few chunks for verification
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(chunk.page_content[:200] + "...")
    
    # Initialize HuggingFace embeddings
    print("\nInitializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save vectorstore
    print("\nCreating vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vectorstore
    print("\nSaving vectorstore...")
    vectorstore.save_local("vectorstore")
    
    return len(chunks)

if __name__ == "__main__":
    try:
        num_chunks = process_documents()
        print(f"\nSuccessfully processed documents and created {num_chunks} chunks!")
        print("\nVectorstore saved to 'vectorstore' directory")
    except Exception as e:
        print(f"\nError: {str(e)}")