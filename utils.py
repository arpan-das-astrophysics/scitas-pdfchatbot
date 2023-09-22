import logging
import os
import re
import shutil
import sys
from typing import List

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    OnlinePDFLoader,
    WebBaseLoader,
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    FETCH_K,
    MAX_TOKENS,
    MODEL,
    PAGE_ICON,
    TEMPERATURE,
    K,
)

# loads environment variables
load_dotenv()

logger = logging.getLogger(APP_NAME)

def authenticate(
    openai_api_key: str, activeloop_token: str, activeloop_org_name: str
) -> None:
    # Validate all credentials are set and correct
    # Check for env variables to enable local dev and deployments with shared credentials
    openai_api_key = (
        openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    activeloop_token = (
        activeloop_token
        or os.environ.get("ACTIVELOOP_TOKEN")
        or st.secrets.get("ACTIVELOOP_TOKEN")
    )
    activeloop_org_name = (
        activeloop_org_name
        or os.environ.get("ACTIVELOOP_ORG_NAME")
        or st.secrets.get("ACTIVELOOP_ORG_NAME")
    )
    if not (openai_api_key and activeloop_token and activeloop_org_name):
        st.session_state["auth_ok"] = False
        st.error("Credentials neither set nor stored", icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = openai_api_key
            openai.Model.list()
            deeplake.exists(
                f"hub://{activeloop_org_name}/SCITASCHAT-Authentication-Check",
                token=activeloop_token,
            )
    except Exception as e:
        logger.error(f"Authentication failed with {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed", icon=PAGE_ICON)
        return
    # store credentials in the session state
    st.session_state["auth_ok"] = True
    st.session_state["openai_api_key"] = openai_api_key
    st.session_state["activeloop_token"] = activeloop_token
    st.session_state["activeloop_org_name"] = activeloop_org_name
    logger.info("Authentification successful!")

def handle_load_error(e: str = None) -> None:
    e = e or "No Loader found for your data source."
    error_msg = f"Failed to load {st.session_state['data_source']} with Error:\n{e}"
    st.error(error_msg, icon=PAGE_ICON)
    logger.info(error_msg)
    st.stop()

def load_data_source(
    data_source: str, chunk_size: int = CHUNK_SIZE
) -> List[Document]:
    # Load the data, only http urls are loaded now

    is_web = data_source.startswith("http")
    is_pdf = data_source.endswith(".pdf")

    loader = None

    if is_web:
        if is_pdf:
            loader = OnlinePDFLoader(data_source)
        else:
            loader = WebBaseLoader(data_source)
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        docs = loader.load_and_split(text_splitter)

        logger.info(f"Loaded: {len(docs)} documents")

        return docs
        
    except Exception as e:
        handle_load_error(e if loader else None)

def clean_data_source_string(data_source_string: str) -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dashed_string = re.sub(r"\W+", "-", data_source_string)
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string

def setup_vector_store(data_source: str, chunk_size: int = CHUNK_SIZE) -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
    )
    data_source_name = clean_data_source_string(data_source)
    dataset_path = f"hub://{st.session_state['activeloop_org_name']}/{data_source_name}-{chunk_size}"
    if deeplake.exists(dataset_path, token=st.session_state["activeloop_token"]):
        with st.spinner("Loading vector store..."):
            logger.info(f"Dataset '{dataset_path}' exists -> loading")
            vector_store = DeepLake(
                dataset_path=dataset_path,
                read_only=True,
                embedding_function=embeddings,
                token=st.session_state["activeloop_token"],
            )
    else:
        with st.spinner("Reading, embedding and uploading data to hub..."):
            logger.info(f"Dataset '{dataset_path}' does not exist -> uploading")
            docs = load_data_source(data_source, chunk_size)
            vector_store = DeepLake.from_documents(
                docs,
                embeddings,
                dataset_path=dataset_path,
                token=st.session_state["activeloop_token"],
            )
    return vector_store

def build_chain(
    data_source: str,
    k: int = K,
    fetch_k: int = FETCH_K,
    chunk_size: int = CHUNK_SIZE,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> ConversationalRetrievalChain:
    # create the langchain that will be called to generate responses
    vector_store = setup_vector_store(data_source, chunk_size)
    retriever = vector_store.as_retriever()
    # Search params "fetch_k" and "k" define how many documents are pulled from the hub
    # and selected after the document matching to build the context
    # that is fed to the model together with your prompt
    search_kwargs = {
        "maximal_marginal_relevance": True,
        "distance_metric": "cos",
        "fetch_k": fetch_k,
        "k": k,
    }
    retriever.search_kwargs.update(search_kwargs)
    model = ChatOpenAI(
        model_name=MODEL,
        temperature=temperature,
        openai_api_key=st.session_state["openai_api_key"],
    )
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        # we limit the maximum number of used tokens
        # to prevent running into the models token limit of 4096
        max_tokens_limit=max_tokens,
    )
    logger.info(f"Data source '{data_source}' is ready to go!")
    return chain

def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        st.session_state["chain"] = build_chain(
            data_source=st.session_state["data_source"],
            k=st.session_state["k"],
            fetch_k=st.session_state["fetch_k"],
            chunk_size=st.session_state["chunk_size"],
            temperature=st.session_state["temperature"],
            max_tokens=st.session_state["max_tokens"],
        )
        st.session_state["chat_history"] = []
    except Exception as e:
        msg = f"Failed to build chain for data source {st.session_state['data_source']} with error: {e}"
        logger.error(msg)
        st.error(msg, icon=PAGE_ICON)

def generate_response(prompt: str) -> str:
    # call the chain to generate responses and add them to the chat history
    with st.spinner("Generating response"), get_openai_callback() as cb:
        response = st.session_state["chain"](
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )
    logger.info(f"Response: '{response}'")
    st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]

