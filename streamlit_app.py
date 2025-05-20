# streamlit run streamlit_app.py
import os
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

import torch
torch.classes.__path__ = []

import streamlit as st
import warnings
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.messages import HumanMessage, AIMessage

# Inject CSS styles
def inject_chat_styles():
    st.markdown("""
        <style>
        body {
            background-color: white !important;
            color: black !important;
        }
        .message {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
            color: black !important;
        }
        .user {
            background-color: #FFFFFF;
            align-self: flex-end;
            border: 1px solid #CCC;
        }
        .bot {
            background-color: #F7F7F7;
            align-self: flex-start;
            border: 1px solid #CCC;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)

# Streamlit layout
st.set_page_config(page_title="LangChain Chatbot", layout="wide")
inject_chat_styles()
st.title("💬 LangChain Chat Interface")

# In-memory store for multiple sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Initialize chain and memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'qa_chain' not in st.session_state:
    with st.spinner("Loading documents and setting up the chain..."):
        documents = []
        data_dir = "data"
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(data_dir, filename))
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever()
        st.session_state.retriever = retriever

        llm = ChatOllama(model="gemma3")

        # Setup RAG with history-aware retriever
        retriever_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        stuff_prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=(
                "Use the following pieces of context to answer the question at the end.\n"
                "If you don't know the answer, just say that you don't know.\n\n"
                "{context}\n\nQuestion: {input}\nHelpful Answer:"
            )
        )

        question_answer_chain = create_stuff_documents_chain(llm, stuff_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        qa_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        st.session_state.qa_chain = qa_chain

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke(
            {
                "input": user_input
            },
            config={"configurable": {"session_id": "default"}}
        )

        response_text = response["answer"]
        st.session_state.chat_history.append((user_input, response_text))

# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for q, a in st.session_state.chat_history:
    st.markdown(f"<div class='message user'><strong>You:</strong><br>{q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='message bot'><strong>Bot:</strong><br>{a}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
