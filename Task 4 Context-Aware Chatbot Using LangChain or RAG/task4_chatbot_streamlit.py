"""
═══════════════════════════════════════════════════════════════════════════
Task 4: RAG Chatbot - Streamlit Web Application
─────────────────────────────────────────────────────────────────────────
DevelopersHub Corporation — AI/ML Engineering Advanced Internship

Run with: streamlit run task4_chatbot_streamlit.py

Features:
  • Interactive chat interface
  • Real-time document retrieval
  • Conversation history tracking
  • Source document display
  • Export chat history
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the RAG Chatbot
from task4_rag_chatbot_core import RAGChatbot, Config

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .bot-message {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #9C27B0;
    }
    
    .source-box {
        background-color: #FFF9C4;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 3px solid #FBC02D;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .sidebar-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Metric styling */
    .metric-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    /* Header styling */
    h1 {
        color: #2C3E50;
        border-bottom: 3px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #34495E;
        margin-top: 1.5rem;
    }
    
    /* Code block styling */
    .code-block {
        background-color: #282C34;
        color: #ABB2BF;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.chatbot_ready = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'stats' not in st.session_state:
        st.session_state.stats = {}
    
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_chatbot(use_pretrained: bool = True):
    """Load and initialize RAG chatbot (cached for performance)."""
    with st.spinner("🔧 Initializing RAG Chatbot... This may take a minute..."):
        try:
            chatbot = RAGChatbot()
            chatbot.setup(use_pretrained=use_pretrained)
            st.success("✅ Chatbot initialized successfully!")
            return chatbot
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {str(e)}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit app."""
    
    # Initialize session state
    initialize_session_state()
    
    # ─────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 RAG-based Context-Aware Chatbot")
        st.markdown("""
        **Powered by LangChain + FAISS + Transformers**
        
        Ask me anything! I'll retrieve relevant documents and provide accurate answers.
        """)
    
    with col2:
        st.image(
            "https://img.icons8.com/color/96/000000/chatbot.png",
            width=80
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Setup section
        st.subheader("🔧 Setup")
        
        if st.button("🚀 Load Chatbot", use_container_width=True):
            with st.spinner("Loading chatbot..."):
                chatbot = load_chatbot(use_pretrained=True)
                if chatbot:
                    st.session_state.chatbot = chatbot
                    st.session_state.setup_complete = True
                    st.rerun()
        
        if st.session_state.setup_complete:
            st.success("✅ Chatbot Ready")
        
        # Settings section
        st.divider()
        st.subheader("⚡ Settings")
        
        config = Config()
        
        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=5,
            value=config.TOP_K_DOCUMENTS,
            help="More documents = slower but potentially better answers"
        )
        
        temperature = st.slider(
            "Temperature (creativity)",
            min_value=0.0,
            max_value=1.0,
            value=config.TEMPERATURE,
            step=0.1,
            help="Higher = more creative, Lower = more consistent"
        )
        
        # Retrieve mode
        retrieval_method = st.selectbox(
            "Retrieval method",
            ["similarity", "mmr"],
            help="MMR avoids redundancy in retrieved documents"
        )
        
        # Stats section
        st.divider()
        st.subheader("📊 Statistics")
        
        if st.session_state.setup_complete and st.session_state.chatbot:
            stats = st.session_state.chatbot.get_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Messages",
                    len(st.session_state.messages),
                    delta=None,
                    help="Total messages in conversation"
                )
            
            with col2:
                st.metric(
                    "Device",
                    stats.get('device', 'CPU'),
                    help="GPU or CPU"
                )
            
            with st.expander("📈 Full Statistics"):
                st.json(stats)
        
        # History section
        st.divider()
        st.subheader("📜 History")
        
        if st.button("💾 Export Conversation", use_container_width=True):
            if st.session_state.messages:
                # Create JSON export
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'messages': st.session_state.messages,
                    'total_messages': len(st.session_state.messages),
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No messages to export")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("History cleared!")
            st.rerun()
        
        # About section
        st.divider()
        st.subheader("ℹ️ About")
        
        st.markdown("""
        **RAG Chatbot v1.0**
        
        Built with:
        - 🦗 LangChain
        - 🔍 FAISS
        - 🤗 HuggingFace
        - 💬 Streamlit
        
        **How it works:**
        1. Your question is converted to embeddings
        2. Similar documents are retrieved from the vector DB
        3. The LLM generates an answer using retrieved context
        4. Memory tracks conversation history
        """)
    
    # ─────────────────────────────────────────────────────────────────────
    # MAIN CHAT AREA
    # ─────────────────────────────────────────────────────────────────────
    
    if not st.session_state.setup_complete:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(
                "👈 **Click 'Load Chatbot' in the sidebar to get started!**",
                icon="ℹ️"
            )
        st.stop()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            for msg in st.session_state.messages:
                if msg['type'] == 'user':
                    st.markdown(f"""
                    <div class='user-message'>
                        <b>👤 You:</b><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif msg['type'] == 'bot':
                    st.markdown(f"""
                    <div class='bot-message'>
                        <b>🤖 Chatbot:</b><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources
                    if msg.get('sources'):
                        st.markdown("<b>📚 Sources:</b>")
                        for i, source in enumerate(msg['sources'], 1):
                            with st.expander(f"Source {i}: {source['metadata'].get('source', 'Unknown')}"):
                                st.write(source['content'])
        else:
            st.info("No messages yet. Start a conversation below!", icon="💬")
    
    # ─────────────────────────────────────────────────────────────────────
    # INPUT & RESPONSE GENERATION
    # ─────────────────────────────────────────────────────────────────────
    
    st.divider()
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "💭 Ask me anything:",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send ➤", use_container_width=True)
    
    # Process user input
    if submit_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Display user message immediately
        st.markdown(f"""
        <div class='user-message'>
            <b>👤 You:</b><br>
            {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate bot response
        with st.spinner("🤔 Thinking..."):
            try:
                chatbot = st.session_state.chatbot
                result = chatbot.ask(user_input, return_sources=True)
                
                bot_response = result.get('answer', 'Error: No response generated')
                sources = result.get('sources', [])
                
                # Add bot message to chat
                st.session_state.messages.append({
                    'type': 'bot',
                    'content': bot_response,
                    'sources': sources,
                    'num_sources': result.get('num_sources', 0),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Display bot response
                st.markdown(f"""
                <div class='bot-message'>
                    <b>🤖 Chatbot:</b><br>
                    {bot_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if sources:
                    st.markdown("<b>📚 Sources Used:</b>")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {source['metadata'].get('source', 'Unknown')}"):
                            st.write(source['content'])
                
                # Rerun to update display
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"📊 Total Messages: {len(st.session_state.messages)}")
    with col2:
        st.caption(f"⏰ Session Started: {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        st.caption("🚀 Task 4 — DevelopersHub Internship")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE DATA & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def create_sample_data():
    """Create sample data directory for testing."""
    import os
    
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    samples = {
        'ai_overview.txt': """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is the simulation of human intelligence by machines.
Machine Learning (ML) is a subset of AI that enables systems to learn from data.
Deep Learning uses neural networks with multiple layers.

Key Technologies:
- Neural Networks
- Transformers
- Reinforcement Learning
- Natural Language Processing

Applications:
- Computer Vision
- Chatbots and NLP
- Autonomous Vehicles
- Medical Diagnosis
- Recommendation Systems
        """,
        
        'python_guide.txt': """
Python for AI/ML

Python is the most popular language for AI and ML development.

Popular Libraries:
1. TensorFlow - Deep learning framework
2. PyTorch - Dynamic neural networks
3. Scikit-learn - Traditional ML algorithms
4. Keras - High-level neural networks API
5. Pandas - Data manipulation
6. NumPy - Numerical computing

Best Practices:
- Use virtual environments
- Write clean, documented code
- Use type hints
- Follow PEP 8 style guide
- Test your code thoroughly
        """,
        
        'nlp_tutorial.txt': """
Natural Language Processing

NLP is a field of AI focused on human language understanding.

Key Tasks:
1. Tokenization - Breaking text into words/phrases
2. Sentiment Analysis - Determining emotional tone
3. Named Entity Recognition - Identifying people, places, organizations
4. Machine Translation - Converting between languages
5. Question Answering - Answering questions about texts

Recent Advances:
- Transformer Models (BERT, GPT)
- Pre-trained Language Models
- Few-shot Learning
- Prompt Engineering
        """
    }
    
    for filename, content in samples.items():
        filepath = data_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
    
    print(f"✅ Created sample data in {data_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
