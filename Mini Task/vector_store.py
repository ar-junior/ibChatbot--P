from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# model
model = ChatGroq(
    model="llama-3.1-8b-instant",
)

# prompt
prompt = PromptTemplate(
    template = " you are a agent",
    input_variables=["topic"]
)

# parsers
parser = StrOutputParser()

# doc loder
loder = PyPDFLoader("path")
doc = loder.lazy_load()

# spliter schema
spliter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

small_doc = spliter.split_documents(doc)

# vector db

vactore_store = FAISS(
    embedding_function= ChatGroq(),
    persist_directory="newdb",
    collection_name="name"
)

vactore_store.aadd_documents(small_doc)

vactore_store.get(include=['embedding','small_doc','metadata'])

vactore_store.similarity_search(
    query="what is this",
    k=2
)

vactore_store.similarity_search_with_score()

vactore_store.update_document(document_id="",document="")

vactore_store.delete(ids="")