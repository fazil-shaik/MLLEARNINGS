from langchain_unstructured import UnstructuredLoader

data = UnstructuredLoader("./demo.txt")

print(data)