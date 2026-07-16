import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

file_paths = [
    "./docloader/demo.txt"
]

loader = UnstructuredLoader(file_paths)

docs = loader.load()

# Convert the documents into a single string
text = "\n".join(doc.page_content for doc in docs)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI summarizer. Summarize the given text in less than 250 characters."
        ),
        (
            "human",
            "{text}"
        ),
    ]
)

model = ChatMistralAI(
    model="mistral-small-2603",
    api_key=api_key
)

prompt = template.format_messages(text=text)

response = model.invoke(prompt)

print(response.content)