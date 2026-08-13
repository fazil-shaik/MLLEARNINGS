from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GOOGLE_API_KEY
from app.rag.retriever import retrieve_context


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def generate_answer(question: str):

    documents = retrieve_context(question)

    if not documents:
        return "I couldn't find relevant information in the documents."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Do not make up information.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content