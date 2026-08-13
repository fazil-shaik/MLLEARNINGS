import os
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.loader.pdf_loader import load_pdf
from app.vectorstore.chroma import add_documents
from app.rag.chain import generate_answer


router = APIRouter()


class QuestionRequest(BaseModel):

    question: str


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename.endswith(".pdf"):
        return {
            "error": "Only PDF files are supported"
        }

    temp_dir = tempfile.mkdtemp()

    file_path = os.path.join(
        temp_dir,
        file.filename
    )

    try:

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # Load and split PDF
        chunks = load_pdf(file_path)

        # Store chunks in Chroma
        add_documents(chunks)

        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks": len(chunks)
        }

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


@router.post("/ask")
async def ask_question(
    request: QuestionRequest
):

    answer = generate_answer(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }