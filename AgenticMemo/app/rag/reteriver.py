from app.vectorstore.chroma import search_documents


def retrieve_context(query: str, k: int = 4):

    documents = search_documents(
        query=query,
        k=k
    )

    return documents