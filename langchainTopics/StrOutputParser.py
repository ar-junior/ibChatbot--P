from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

temp1 = PromptTemplate(
    template="give me this {topic} detaild report, and 5 line summary",
    input_variables=["topic"]
)

temp2 = PromptTemplate(
    template="give me the 5 line summary with indexing from this {text}",
    input_variables=["text"]
)

pars = StrOutputParser()

user = input("topic :")

chain = temp1 | model | pars | temp2 | model | pars

result = chain.invoke({"topic": user})

print(result)

print(result)