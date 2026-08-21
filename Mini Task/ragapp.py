from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ----------------- INDEXING

# data loader -> document
loader = TextLoader("text.txt")
# loder = PyPDFLoader("sample.pdf")
doc = loader.load()


# splitter schima
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],
    chunk_size = 200,
    chunk_overlap = 0
)
# document splitter -> chunk
docs = splitter.split_documents(doc)


# embedding+store 
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    collection_name = "Records"
)

# ----------------- RETRIEVAL

query = "Where is the medical room located?"

# retriever schima 
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2,
    }
)

relevant_doc = retriever.invoke(query)

# ----------------- AUGMENTATION

context = "\n\n".join(doc.page_content for doc in relevant_doc)

prompt = PromptTemplate(
    template="""
You are an expert document analysis assistant.
Your task is to answer the user's question strictly using the provided document context.
Rules:
1. Use only the information available in the context.
2. Do not add external knowledge.
3. If the answer is not available, say:
   "The provided document does not contain this information."
4. If the answer spans multiple parts of the context, combine them into one complete answer.
5. Keep the response concise and factual.
6. If numbers, dates, names, or statistics are present, include them exactly as written.
Document Context:
{context}
User Question:
{question}
Answer:
""",
input_variables=["context","question"]
)

# ----------------- GENERATION

model=ChatGroq(
    model = "llama-3.3-70b-versatile"
)

parser = StrOutputParser()

chain = prompt | model | parser

answer = chain.invoke({
    "context":context,
    "question":query
})

print(f"\n⦾ USER: {query}")
print(f"⦿ BOT : {answer}\n")