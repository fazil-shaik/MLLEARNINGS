# from pypdf import PdfReader


# def load_pdf(file_path):
#     """
#     Load a PDF file and return its text content.

#     Args:
#         file_path (str): The path to the PDF file."""
#     reader = PdfReader(file_path)
#     text_content = ""
#     for page in reader.pages:
#         text_content += page.extract_text() + "\n"
#     return text_content

# res = load_pdf("../data/documents/AIPROOF.pdf")

# print(res[:500])  # Print the first 500 characters of the extracted text

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    return chunks