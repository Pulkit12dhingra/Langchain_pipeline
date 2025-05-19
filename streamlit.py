# to run this code use command streamlit run streamlit.py ( file name can be anything) in the terminal

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

# Title
st.title("LangChain RAG Chatbot")

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

# Input
user_input = st.text_input("Ask a question:", key="input")

# On submit
if user_input:
    with st.spinner("Generating answer..."):
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
