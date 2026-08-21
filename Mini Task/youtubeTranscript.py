from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import strin
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv

# step 1 - Indexing (document ingestion)

yt = YouTubeTranscriptApi()
video_id = "LPZh9BOjkQs"
try:
    transcript_list = yt.fetch(
        video_id,
        languages = ["en"]
    )
    transcript = " ".join(chunk.text for chunk in transcript_list)  
except TranscriptsDisabled:
    print("Transcript is disabled.")



split = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)
doc = split.split_documents(transcript)

model = ChatGroq()

vactor_store = FAISS.from_documents(
    embedding= model,
    documents= doc
)


retriver = vactor_store.as_retriever(search_type = "mmr",language = "en" )

query = "who is virat khohli"

result = retriver.invoke(query,kwargs=2)

print(result)