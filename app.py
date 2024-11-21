import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Restaurant Recipes RAG Chatbot",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stExpander {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

class ChatbotComponents:
    """Class to manage chatbot components initialization and operations"""
    
    @staticmethod
    @st.cache_resource
    def initialize_components() -> Optional[RetrievalQA]:
        """Initialize embedding model, vectorstore, and language model"""
        try:
            # Initialize embedding model
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load vectorstore
            # Update this part in the initialize_components method
            vectorstore_path = "vectorstore"  # Relative path
            
            if not os.path.exists(vectorstore_path):
                logger.error(f"Vectorstore not found at: {vectorstore_path}")
                return None
                
            vectorstore = FAISS.load_local(
                    vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                    )
            # Initialize Anthropic Claude model
            llm = ChatAnthropic(
                model="claude-2.1",
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3,
                max_tokens=1000
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            logger.info("Successfully initialized all components")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            st.error("Failed to initialize chatbot components. Please check the logs.")
            return None

def display_chat_interface():
    """Display the chat interface and handle user interactions"""
    st.title("🍳 Restaurant Recipes RAG Chatbot")
    
    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    qa_chain = ChatbotComponents.initialize_components()
    
    if qa_chain is None:
        st.error("Failed to initialize the chatbot. Please check the configuration.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask a question about Restaurant Recipes...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke(prompt)
                
                # Display the response
                st.markdown(response['result'])
                
                # Show sources if available
                if response.get('source_documents'):
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(response['source_documents']):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['result']
                })
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error("Sorry, I encountered an error while generating the response. Please try again.")

def main():
    """Main function to run the Streamlit app"""
    try:
        # Add a sidebar with additional information
        with st.sidebar:
            st.title("About")
            st.markdown("""
            This chatbot uses RAG (Retrieval-Augmented Generation) to answer 
            questions about restaurant recipes. It combines:
            
            - 🤖 Language Model: ANTHROPIC
            - 📚 Embeddings: all-MiniLM-L6-v2
            - 🔍 Vector Store: FAISS
            """)
            
            # Add clear chat button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.experimental_rerun()
        
        # Display main chat interface
        display_chat_interface()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()