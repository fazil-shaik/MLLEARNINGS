from langchain_unstructured import UnstructuredLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter


data = UnstructuredLoader("./demo.txt")


docs = data.load()

splitterdata =  CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    separator="",
    chunk_size = 1000,
    chunk_overlap = 1,
)



chunks = splitterdata.split_documents(docs)

print(len(chunks))

print(chunks[1:5])

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
