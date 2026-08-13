from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_PATH, COLLECTION_NAME
from app.embeddings.embedder import get_embeddings


def get_vectorstore():

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


def add_documents(documents: list[Document]):

    vectorstore = get_vectorstore()

    vectorstore.add_documents(documents)

    return vectorstore


def search_documents(query: str, k: int = 4):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results