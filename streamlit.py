import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama

st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.markdown("""
    <style>
    .message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot {
        background-color: #F1F0F0;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 LangChain Chat Interface")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'qa_chain' not in st.session_state:
    with st.spinner("Loading documents and setting up the chain..."):
        # Load documents
        documents = []
        data_dir = "data"
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Embedding
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever()

        # LLM and memory
        llm = ChatOllama(model="gemma3")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Build chain
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        st.session_state.qa_chain = qa_chain

# Chat input box
user_input = st.chat_input("Type your message here...")

# On submit
if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

# Display chat history as styled chat bubbles
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"<div class='message user'><strong>You:</strong><br>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='message bot'><strong>Bot:</strong><br>{a}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
