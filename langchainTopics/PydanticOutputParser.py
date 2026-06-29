from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage



load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

schema = [
    ResponseSchema(name = "fact_1", description = "fact 1 about the topic"),
    ResponseSchema(name = "fact_2", description = "fact 2 about the topic"),
    ResponseSchema(name = "fact_3", description = "fact 3 about the topic")
]

pars = StructuredOutputParser.from_response_schemas(schema)

temp = PromptTemplate(
    template="give 3 fact about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": pars.get_format_instructions()}
)

prompt = temp.invoke({"topic": "Virat kohli"})
result = model.invoke(prompt)
final_result = pars.parse(result.content)
# chain = temp | model | pars
# final_result = chain.invoke({})
print(final_result)