from langchain_groq import ChatGroq
from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

loader = TextLoader("text.txt")

docs = loader.load()

spliter  = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

spliter2 = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =0
)

test_spliter = spliter2.split_documents(docs)
print(test_spliter[0].page_content)

# result = spliter.split_documents(docs)

# print(len(result))