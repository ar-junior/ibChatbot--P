# import AI model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# get API key
load_dotenv()

# model 
emd = OpenAIEmbeddings(model="gpt-4", dimensions=50)

# document, Data
doc = [
    "Virat Kohli, an Indian professional cricketer. He is the current captain of the Indian national cricket team.",
    "The Indian Premier League (IPL) 2026 is yet to be confirmed, but it's expected to take place in the middle of the year, likely around April-May, in India. Details will be announced later by the BCCI.",
    "Narendra Modi is the former Prime Minister of India, serving from 2014 to 2024. He is a member of the Bharatiya Janata Party (BJP). Before becoming PM, he was the Chief Minister of Gujarat."
]

# Question
query = "who is virat"

# embedding apply
doc_emd = emd.aembed_documents(doc)
qry_emd = emd.aembed_query(query)

# semantic search

# higher similarity vector extract

