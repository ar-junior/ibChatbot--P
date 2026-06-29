from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated

load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant"
)


class Cricketer(BaseModel):
    name : str = Field(
        description="name of the Cricketer"
    )
    playstyle : list[str] = Field(
        description="cricketing skills"
    )
    # email : EmailStr
    tournaments : str = Field(
        description="Formats or tournaments played"
    )
    highest_score : Optional[int] = Field(gt=0 , description="Highest individual score made by the cricketer"
    )

model_schema = model.with_structured_output(Cricketer)

result = model_schema.invoke("Virat Kohli is an Indian cricketer and one of the greatest batsmen in modern cricket. His batting skills include cover drive, straight drive, flick shot, and chase-master innings under pressure. He has represented India in ODI, T20I, Test cricket, ICC Cricket World Cup, ICC Champions Trophy, and the Indian Premier League (IPL). His highest individual score in international cricket is 254 runs.")

student_dict = dict(result)
student_json = result.model_dump_json()
print(student_json)