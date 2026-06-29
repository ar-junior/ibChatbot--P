from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage



load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

pars = JsonOutputParser()

temp = PromptTemplate(
    template="give me the name, age and city of the fictinoal person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": pars.get_format_instructions()}
)

prompt = temp.invoke({})
result = model.invoke(prompt)
final_result = pars.parse(result.content)
# chain = temp | model | pars
# final_result = chain.invoke({})
print(final_result)
