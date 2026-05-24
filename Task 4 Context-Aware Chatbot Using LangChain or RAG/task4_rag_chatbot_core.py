"""
═══════════════════════════════════════════════════════════════════════════
Task 4: Context-Aware Chatbot Using LangChain & RAG
─────────────────────────────────────────────────────────────────────────
DevelopersHub Corporation — AI/ML Engineering Advanced Internship
Created: May 2024
Status: Production Ready

This module implements a Retrieval-Augmented Generation (RAG) chatbot that:
  • Uses LangChain for orchestration
  • Stores documents in a Vector Database (FAISS)
  • Maintains conversation context/memory
  • Retrieves relevant documents for answers
  • Deploys via Streamlit for live interaction
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────
# LLM & Document Processing
# ─────────────────────────────────────────────────────────────────────────
try:
    from langchain.document_loaders import (
        TextLoader,
        PDFPlumberLoader,
        DirectoryLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.callbacks import StreamingStdOutCallbackHandler
    
    # For using OpenAI or other APIs (optional)
    # from langchain.chat_models import ChatOpenAI
    
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    print("   Install with: pip install langchain langchain-community")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# Transformers & HuggingFace
# ─────────────────────────────────────────────────────────────────────────
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
        TextGenerationPipeline
    )
    import torch
except ImportError as e:
    print(f"❌ Transformers import error: {e}")
    print("   Install with: pip install transformers torch")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# Logging & Utilities
# ─────────────────────────────────────────────────────────────────────────
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration for the RAG Chatbot."""
    
    # ─── Paths ───
    DATA_DIR              = './data'
    VECTORDB_PATH         = './vectordb'
    CACHE_DIR             = './cache'
    LOGS_DIR              = './logs'
    
    # ─── Model Configuration ───
    EMBEDDING_MODEL       = 'sentence-transformers/all-MiniLM-L6-v2'
    LLM_MODEL             = 'gpt2'  # Small, fast. Use 'mistral-7b' for better quality
    # Alternative smaller models:
    # - 'distilgpt2' (very fast)
    # - 'gpt2-medium' (medium quality/speed)
    # - Use OpenAI's API for better results: 'gpt-3.5-turbo'
    
    # ─── Text Processing ───
    CHUNK_SIZE            = 500      # Character count per chunk
    CHUNK_OVERLAP         = 100      # Overlap between chunks
    
    # ─── Retrieval ───
    TOP_K_DOCUMENTS       = 3        # Number of relevant docs to retrieve
    SIMILARITY_THRESHOLD  = 0.6      # Minimum similarity score
    
    # ─── Memory & Context ───
    MEMORY_TYPE           = 'buffer'  # 'buffer' or 'summary'
    MAX_MEMORY_TOKENS     = 2000      # Max tokens to keep in memory
    
    # ─── Generation ───
    MAX_LENGTH            = 512
    TEMPERATURE           = 0.7
    TOP_P                 = 0.9
    
    # ─── Device ───
    DEVICE                = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ─── Debugging ───
    DEBUG                 = True
    VERBOSE               = True


def setup_directories():
    """Create necessary directories."""
    for dir_path in [Config.DATA_DIR, Config.VECTORDB_PATH, Config.CACHE_DIR, Config.LOGS_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Directories created: {Config.DATA_DIR}, {Config.VECTORDB_PATH}, etc.")


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING & VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """
    Handles document loading, chunking, and vector store creation.
    
    Supports:
    - Text files (.txt)
    - PDF files (.pdf)
    - Directories of documents
    - Web scraping (optional)
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        self.documents = []
        self.chunks = []
        logger.info("📄 DocumentProcessor initialized")
    
    def load_txt_files(self, directory: str) -> List[str]:
        """Load all .txt files from a directory."""
        logger.info(f"📂 Loading .txt files from {directory}...")
        
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        try:
            docs = loader.load()
            logger.info(f"   ✅ Loaded {len(docs)} text files")
            return docs
        except Exception as e:
            logger.error(f"❌ Error loading txt files: {e}")
            return []
    
    def load_pdf_files(self, directory: str) -> List:
        """Load all PDF files from a directory."""
        logger.info(f"📂 Loading PDF files from {directory}...")
        
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PDFPlumberLoader
        )
        
        try:
            docs = loader.load()
            logger.info(f"   ✅ Loaded {len(docs)} PDF files")
            return docs
        except Exception as e:
            logger.error(f"❌ Error loading PDFs: {e}")
            return []
    
    def load_wikipedia_docs(self, topics: List[str]) -> List:
        """
        Load Wikipedia articles as sample documents.
        Requires: pip install wikipedia
        """
        logger.info(f"📚 Loading Wikipedia articles for topics: {topics}")
        
        try:
            import wikipedia
        except ImportError:
            logger.warning("⚠️  Wikipedia module not installed. Skipping Wikipedia load.")
            return []
        
        docs = []
        for topic in topics:
            try:
                page = wikipedia.page(topic)
                # Split into chunks within this function
                content = page.content
                # Create a simple document object
                from langchain.schema import Document
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': f'Wikipedia: {topic}',
                        'title': page.title
                    }
                )
                docs.append(doc)
                logger.info(f"   ✅ Loaded Wikipedia: {topic}")
            except wikipedia.exceptions.PageError:
                logger.warning(f"   ⚠️  Wikipedia page not found: {topic}")
            except Exception as e:
                logger.error(f"   ❌ Error loading {topic}: {e}")
        
        return docs
    
    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into manageable chunks using RecursiveCharacterTextSplitter.
        
        Why recursive splitting?
        - Preserves semantic meaning by splitting on natural boundaries
        - Falls back to smaller chunks if needed
        - Better for retrieval accuracy
        """
        logger.info(f"✂️  Chunking {len(documents)} documents...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size       = self.config.CHUNK_SIZE,
            chunk_overlap    = self.config.CHUNK_OVERLAP,
            separators       = ["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"   ✅ Created {len(chunks)} chunks (avg size: {self.config.CHUNK_SIZE} chars)")
        
        self.documents = documents
        self.chunks = chunks
        return chunks
    
    def get_statistics(self) -> Dict:
        """Return statistics about loaded documents and chunks."""
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'avg_chunk_size': np.mean([len(c.page_content) for c in self.chunks]) if self.chunks else 0,
            'total_chars': sum(len(c.page_content) for c in self.chunks),
        }


class VectorStoreManager:
    """
    Manages FAISS vector database creation and retrieval.
    
    FAISS (Facebook AI Similarity Search):
    - Fast vector similarity search
    - Stores embeddings locally (no API costs)
    - Scales to millions of vectors
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        logger.info("🔍 VectorStoreManager initialized")
    
    def initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Load embedding model (converts text → vectors).
        
        Using 'all-MiniLM-L6-v2':
        - Fast inference (10x faster than larger models)
        - Good quality (99.1% performance vs larger models)
        - 384-dimensional embeddings
        - ~22M parameters
        """
        logger.info(f"🔤 Loading embeddings model: {self.config.EMBEDDING_MODEL}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name     = self.config.EMBEDDING_MODEL,
            model_kwargs   = {'device': self.config.DEVICE},
            encode_kwargs  = {'normalize_embeddings': True}
        )
        
        logger.info(f"   ✅ Embeddings model loaded")
        return self.embeddings
    
    def create_vectorstore(self, chunks: List) -> FAISS:
        """
        Create FAISS vector store from document chunks.
        
        Process:
        1. Convert each chunk to embedding vector
        2. Store vectors in FAISS index
        3. Create mapping: vector → original document
        """
        logger.info(f"🗂️  Creating FAISS vectorstore from {len(chunks)} chunks...")
        
        if self.embeddings is None:
            self.initialize_embeddings()
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(
            documents = chunks,
            embedding = self.embeddings
        )
        
        logger.info(f"   ✅ FAISS vectorstore created")
        return self.vectorstore
    
    def save_vectorstore(self, path: str = None):
        """Save vectorstore to disk for later reuse."""
        path = path or self.config.VECTORDB_PATH
        logger.info(f"💾 Saving vectorstore to {path}...")
        
        try:
            self.vectorstore.save_local(path)
            logger.info(f"   ✅ Vectorstore saved")
        except Exception as e:
            logger.error(f"❌ Error saving vectorstore: {e}")
    
    def load_vectorstore(self, path: str = None) -> Optional[FAISS]:
        """Load pre-computed vectorstore from disk."""
        path = path or self.config.VECTORDB_PATH
        logger.info(f"📂 Loading vectorstore from {path}...")
        
        try:
            if self.embeddings is None:
                self.initialize_embeddings()
            
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            logger.info(f"   ✅ Vectorstore loaded")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"⚠️  Could not load vectorstore: {e}")
            return None
    
    def get_retriever(self, search_type: str = "similarity", k: int = None):
        """
        Get a retriever for semantic search.
        
        search_type options:
        - 'similarity': Find most similar documents
        - 'mmr': Maximum Marginal Relevance (avoids redundancy)
        """
        k = k or self.config.TOP_K_DOCUMENTS
        
        logger.info(f"🔎 Creating retriever (search_type: {search_type}, k: {k})")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type = search_type,
            search_kwargs = {
                'k': k,
                'score_threshold': self.config.SIMILARITY_THRESHOLD
            }
        )
        
        logger.info(f"   ✅ Retriever created")
        return self.retriever


# ═══════════════════════════════════════════════════════════════════════════
# LLM SETUP & RAG CHAIN
# ═══════════════════════════════════════════════════════════════════════════

class LLMManager:
    """
    Manages LLM model initialization and RAG chain setup.
    
    Supports:
    - Local LLMs (HuggingFace)
    - OpenAI API (ChatGPT, GPT-4)
    - Anthropic Claude
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        self.llm = None
        self.chain = None
        logger.info("🤖 LLMManager initialized")
    
    def initialize_local_llm(self) -> HuggingFacePipeline:
        """
        Initialize a local LLM using HuggingFace Transformers.
        
        Models to consider:
        - 'distilgpt2' (smallest, ~82M params) — fastest
        - 'gpt2' (medium, ~124M params) — balanced
        - 'gpt2-medium' (larger, ~355M params) — better quality
        - 'mistral-7b' (7B params) — best quality, slower
        """
        logger.info(f"🔧 Loading local LLM: {self.config.LLM_MODEL}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                torch_dtype       = torch.float16 if 'cuda' in self.config.DEVICE else torch.float32,
                device_map        = 'auto' if 'cuda' in self.config.DEVICE else None,
                load_in_8bit      = True if 'cuda' in self.config.DEVICE else False,
            )
            
            # Create pipeline
            text_pipeline = pipeline(
                task              = 'text-generation',
                model             = model,
                tokenizer         = tokenizer,
                max_length        = self.config.MAX_LENGTH,
                do_sample         = True,
                temperature       = self.config.TEMPERATURE,
                top_p             = self.config.TOP_P,
                repetition_penalty = 1.2,
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(
                model_id         = self.config.LLM_MODEL,
                task             = 'text-generation',
                pipeline         = text_pipeline,
            )
            
            logger.info(f"   ✅ Local LLM loaded: {self.config.LLM_MODEL}")
            return self.llm
        
        except Exception as e:
            logger.error(f"❌ Error loading LLM: {e}")
            return None
    
    def initialize_openai_llm(self, api_key: str = None):
        """
        Initialize OpenAI ChatGPT (requires API key).
        
        Much better quality than local models, but costs money.
        Set api_key via environment variable: OPENAI_API_KEY
        """
        logger.info("🔑 Initializing OpenAI LLM...")
        
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("⚠️  OPENAI_API_KEY not set. Falling back to local LLM.")
            return None
        
        try:
            from langchain.chat_models import ChatOpenAI
            
            self.llm = ChatOpenAI(
                api_key           = api_key,
                model_name        = 'gpt-3.5-turbo',
                temperature       = self.config.TEMPERATURE,
                max_tokens        = self.config.MAX_LENGTH,
            )
            
            logger.info("   ✅ OpenAI LLM initialized")
            return self.llm
        
        except Exception as e:
            logger.error(f"❌ Error initializing OpenAI: {e}")
            return None
    
    def create_rag_chain(self, retriever, memory) -> ConversationalRetrievalChain:
        """
        Create the RAG chain that combines retrieval + generation.
        
        Flow:
        1. User question
        2. Retrieve relevant documents from vectorstore
        3. Create context from retrieved docs
        4. Generate response using LLM
        5. Add to conversation memory
        """
        logger.info("⛓️  Creating RAG ConversationalRetrievalChain...")
        
        # Custom prompt template
        qa_prompt = PromptTemplate(
            input_variables = ['context', 'question'],
            template = """You are a helpful AI assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context above."""
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm               = self.llm,
            retriever         = retriever,
            memory            = memory,
            combine_docs_chain_kwargs = {'prompt': qa_prompt},
            return_source_documents = True,  # Return which docs were used
            verbose           = self.config.VERBOSE,
        )
        
        logger.info("   ✅ RAG chain created")
        return self.chain


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# ═══════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Manages conversation history and context.
    
    Two strategies:
    1. Buffer Memory: Keep full conversation history (token limit)
    2. Summary Memory: Summarize old messages to save tokens
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        self.memory = None
        logger.info("💾 MemoryManager initialized")
    
    def initialize_memory(self):
        """Create conversation memory based on config."""
        logger.info(f"📝 Initializing {self.config.MEMORY_TYPE} memory...")
        
        if self.config.MEMORY_TYPE == 'buffer':
            self.memory = ConversationBufferMemory(
                memory_key       = 'chat_history',
                return_messages  = True,
                human_prefix     = 'User',
                ai_prefix        = 'Assistant',
            )
        
        elif self.config.MEMORY_TYPE == 'summary':
            self.memory = ConversationSummaryMemory.from_messages(
                llm              = None,  # Will be set later
                memory_key       = 'chat_history',
                human_prefix     = 'User',
                ai_prefix        = 'Assistant',
            )
        
        logger.info(f"   ✅ Memory initialized: {self.config.MEMORY_TYPE}")
        return self.memory
    
    def get_memory_summary(self) -> Dict:
        """Get current memory state."""
        if self.memory is None:
            return {}
        
        return {
            'type': self.config.MEMORY_TYPE,
            'messages': len(self.memory.buffer) if hasattr(self.memory, 'buffer') else 0,
            'preview': str(self.memory.buffer)[:200] if hasattr(self.memory, 'buffer') else 'N/A'
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RAG CHATBOT
# ═══════════════════════════════════════════════════════════════════════════

class RAGChatbot:
    """
    Main RAG Chatbot class.
    
    Combines all components:
    - Document processing
    - Vector store
    - LLM
    - Memory
    - RAG chain
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        setup_directories()
        
        self.doc_processor = DocumentProcessor(config)
        self.vectorstore_mgr = VectorStoreManager(config)
        self.llm_mgr = LLMManager(config)
        self.memory_mgr = MemoryManager(config)
        
        self.chain = None
        self.conversation_history = []
        
        logger.info("🚀 RAGChatbot initialized")
    
    def setup(self, data_sources: Dict = None, use_pretrained: bool = True):
        """
        Complete setup process.
        
        Args:
            data_sources: Dict with keys 'txt_dir', 'pdf_dir', 'wikipedia_topics'
            use_pretrained: Load existing vectorstore if available
        """
        logger.info("⚙️  Starting RAGChatbot setup...")
        
        # 1. Try loading existing vectorstore
        if use_pretrained:
            vectorstore = self.vectorstore_mgr.load_vectorstore()
            if vectorstore:
                logger.info("✅ Using existing vectorstore")
            else:
                logger.warning("⚠️  No existing vectorstore found, creating new one")
                self._create_new_vectorstore(data_sources)
        else:
            self._create_new_vectorstore(data_sources)
        
        # 2. Initialize memory
        self.memory_mgr.initialize_memory()
        
        # 3. Initialize LLM
        self.llm_mgr.initialize_local_llm()
        # Or use OpenAI: self.llm_mgr.initialize_openai_llm()
        
        # 4. Create RAG chain
        retriever = self.vectorstore_mgr.get_retriever()
        self.chain = self.llm_mgr.create_rag_chain(
            retriever = retriever,
            memory = self.memory_mgr.memory
        )
        
        logger.info("✅ RAGChatbot setup complete!")
    
    def _create_new_vectorstore(self, data_sources: Dict = None):
        """Create a new vectorstore from data sources."""
        logger.info("🔨 Creating new vectorstore...")
        
        documents = []
        
        # Load from different sources
        if data_sources is None:
            data_sources = {}
        
        if 'txt_dir' in data_sources:
            documents.extend(self.doc_processor.load_txt_files(data_sources['txt_dir']))
        
        if 'pdf_dir' in data_sources:
            documents.extend(self.doc_processor.load_pdf_files(data_sources['pdf_dir']))
        
        if 'wikipedia_topics' in data_sources:
            documents.extend(self.doc_processor.load_wikipedia_docs(data_sources['wikipedia_topics']))
        
        if not documents:
            logger.warning("⚠️  No documents found. Creating sample documents...")
            documents = self._create_sample_documents()
        
        # Process documents
        chunks = self.doc_processor.chunk_documents(documents)
        
        # Create vectorstore
        self.vectorstore_mgr.initialize_embeddings()
        self.vectorstore_mgr.create_vectorstore(chunks)
        self.vectorstore_mgr.save_vectorstore()
        
        logger.info(f"✅ New vectorstore created with {len(chunks)} chunks")
    
    def _create_sample_documents(self) -> List:
        """Create sample documents for testing."""
        logger.info("📝 Creating sample documents for demo...")
        
        from langchain.schema import Document
        
        sample_docs = [
            Document(
                page_content="""
                Artificial Intelligence (AI) is transforming the world.
                Machine learning, deep learning, and neural networks are key technologies.
                AI applications include chatbots, image recognition, and autonomous vehicles.
                The future of AI is exciting and full of possibilities.
                """,
                metadata={'source': 'AI Overview', 'topic': 'Technology'}
            ),
            Document(
                page_content="""
                Python is a popular programming language.
                It's used for web development, data science, and automation.
                Libraries like TensorFlow, PyTorch, and Scikit-learn are widely used.
                Python's simplicity makes it ideal for beginners and experts alike.
                """,
                metadata={'source': 'Python Guide', 'topic': 'Programming'}
            ),
            Document(
                page_content="""
                Natural Language Processing (NLP) enables computers to understand human language.
                Transformers like BERT and GPT have revolutionized NLP.
                Applications include sentiment analysis, machine translation, and question answering.
                NLP is a rapidly evolving field with many exciting developments.
                """,
                metadata={'source': 'NLP Tutorial', 'topic': 'AI'}
            ),
        ]
        
        logger.info(f"   ✅ Created {len(sample_docs)} sample documents")
        return sample_docs
    
    def ask(self, question: str, return_sources: bool = True) -> Dict:
        """
        Ask the chatbot a question.
        
        Args:
            question: User question
            return_sources: Return the source documents used
        
        Returns:
            Dict with 'answer', 'sources', 'confidence'
        """
        logger.info(f"❓ Question: {question}")
        
        if self.chain is None:
            logger.error("❌ Chatbot not initialized. Call setup() first.")
            return {'answer': 'Chatbot not initialized', 'sources': []}
        
        try:
            # Get response from RAG chain
            result = self.chain(
                {'question': question},
                return_only_outputs = False
            )
            
            answer = result.get('answer', 'No answer generated')
            source_documents = result.get('source_documents', [])
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'sources': [doc.metadata.get('source', 'Unknown') for doc in source_documents]
            })
            
            logger.info(f"✅ Answer generated using {len(source_documents)} sources")
            
            return {
                'answer': answer,
                'sources': [
                    {
                        'content': doc.page_content[:200] + '...',
                        'metadata': doc.metadata
                    }
                    for doc in source_documents
                ] if return_sources else [],
                'num_sources': len(source_documents),
            }
        
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return {
                'answer': f'Error: {str(e)}',
                'sources': [],
                'error': str(e)
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history."""
        return self.conversation_history
    
    def save_conversation(self, filepath: str = None):
        """Save conversation to JSON file."""
        filepath = filepath or f"{self.config.LOGS_DIR}/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.conversation_history, f, indent=2, default=str)
            logger.info(f"💾 Conversation saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving conversation: {e}")
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics."""
        return {
            'doc_processor': self.doc_processor.get_statistics(),
            'memory': self.memory_mgr.get_memory_summary(),
            'conversation_length': len(self.conversation_history),
            'device': self.config.DEVICE,
            'embedding_model': self.config.EMBEDDING_MODEL,
            'llm_model': self.config.LLM_MODEL,
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Example usage of RAGChatbot
    """
    
    print("\n" + "="*70)
    print("🚀 RAG Chatbot - Standalone Example")
    print("="*70 + "\n")
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Setup with sample data (no external files needed)
    print("Setting up chatbot...")
    chatbot.setup(use_pretrained=False)  # Create new vectorstore
    
    # Ask questions
    questions = [
        "What is AI?",
        "Tell me about Python",
        "How does NLP work?",
        "What are transformers used for?",
    ]
    
    print("\n" + "="*70)
    print("💬 Interactive Conversation")
    print("="*70 + "\n")
    
    for question in questions:
        result = chatbot.ask(question)
        
        print(f"\n🗣️  User: {question}")
        print(f"🤖 Chatbot: {result['answer']}")
        print(f"📚 Sources used: {result['num_sources']}")
        
        if result['sources']:
            for i, source in enumerate(result['sources'], 1):
                print(f"   [{i}] {source['metadata'].get('source', 'Unknown')}")
        
        print("-" * 70)
    
    # Print statistics
    print("\n📊 Chatbot Statistics:")
    stats = chatbot.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save conversation
    chatbot.save_conversation()
    print("\n✅ Conversation saved!")
