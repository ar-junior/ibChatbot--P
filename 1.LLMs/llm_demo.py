from langchain_groq import groq
from dotenv import load_dotenv

load_dotenv()

llm = groq(model="llama-3.1-8b-instant")

ans = llm.invoke("who is virat kohli")

print(ans)