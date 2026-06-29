from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage



load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(gt=18, description="age of the person")
    city : str = Field (description="name of the city the person belongs to")

pars = PydanticOutputParser(pydantic_object=person)


temp = PromptTemplate(
    template="give name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": pars.get_format_instructions()}
)

prompt = temp.invoke({"place": "indian"})
result = model.invoke(prompt)

final_result = pars.parse(result.content)

# chain = temp | model | pars
# final_result = chain.invoke({})
print(prompt)