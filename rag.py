# All imports moved to the top
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This is an example of a list that can be used as a data
docs = [
    "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "The capital of France is Paris.",
    "LangChain is a framework for developing applications powered by language models.",
]
documents = [Document(page_content=doc) for doc in docs]

# We will use a pdf as a data store for looking up additional info
file_path = "data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Creates a text splitter and uses that to split the pdf into a list of Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
documents = text_splitter.split_documents(docs)

# Create an embeddings object, and save the Documents into a VectorDB using.
# Also create a retriever for the VectorDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever()

# Create a model and a chain
chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="mistral"), retriever=retriever, return_source_documents=True
)

query = "who is the number of the employees in Nike"
result = chain.invoke({"query": query})

print("Answer:", result["result"])
