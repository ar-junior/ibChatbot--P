# from langchain_groq import ChatGroq
# from langchain_community.retrievers import WikipediaRetriever
# from langchain_community.vectorstores import FAISS

# retriver = WikipediaRetriever(top_k_results=2, lang="en")

# qurey = "who is virat kohli"

# doc = retriver.invoke(qurey)


# vector_store = FAISS.from_documents(
#     documents=doc,
#     embedding="model"
# )

# result = vector_store.as_retriever(
#     search_type = "mmr",
#     search_kwargs={"k": 2}
# )

# qurey = "who is virar"

# final = result.invoke(qurey)


# # for i, docs in enumerate(doc):
# #     print(f"resilt{i}\n\n{docs.page_content}")





from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS



load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant"
)

loder = PyPDFLoader(
    "path"
)

doc = loder.load()

split = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

samll_doc = split.split_documents(doc)


embeding = FAISS.from_documents(
    embedding="lama",
    documents=samll_doc
)

restivel = embeding.as_retriever(restivel_type = "mmr",)

qurey = " "

result = restivel.invoke(qurey)