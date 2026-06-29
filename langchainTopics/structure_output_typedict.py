from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0.5
)

class reviewSimple(TypedDict):
    summary : str
    sentiment : str


class reviewAnnotated(TypedDict):
    key_thems : Annotated[list[str], "write down all the key themes discusses in a review in a list"]
    summary : Annotated[str , "give topic summary"]
    sentiment : Annotated[str , "return sentiment of the review either positive or nagetive"]
    pros : Annotated[Optional[list[str]],"write down all the pros inside the list"]
    cons : Annotated[Literal["pos","neg"],"write down all the cons inside the list"]

structured_model = model.with_structured_output(reviewAnnotated)

result = structured_model.invoke("prompt")

print(result["summary"])