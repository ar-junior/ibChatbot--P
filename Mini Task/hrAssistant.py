from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# API key
load_dotenv()

# model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 1.5
)

# Prompt schema
prompt = PromptTemplate(
    template= " you are a helpfull HR assistant{msg}",
    input_variables=["msg"]

)

# output parsers
parser = StrOutputParser()

# user input 
user = input("ask...")
print(f"User : {user}")

# final 
chain = prompt | model | parser

result = chain.invoke(
    {"msg":"user"}
)

print(f"Bot : {result}")
