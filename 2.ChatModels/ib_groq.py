# import AI model
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Get API key
load_dotenv()

# Model
model = ChatGroq(model="llama-3.1-8b-instant", temperature=1.2)

# prompt structure
template = PromptTemplate(
    input_variables=["input"],
    template="""
You are a helpful AI assistant.
Rules:
- Give short and meaningful answers.
- Use simple English.
- Answer directly.
- Keep responses within 2-4 sentences.
- Avoid unnecessary details.
- If the answer requires explanation, keep it concise.
User Question: {input}
"""
)

print("\nAsk Questions!!")    # chatbot hader

while True:   # input/output loop
    
    user = input("⦾ User: ")    # user input
    
    if user.lower()=="exit":    # exit from chat
        print("chatbot ended...")
        break
    try:
        tmp_ans = template.format(input=user)   # send input to template
        bot = model.invoke(tmp_ans)
        print(f"⦿ Bot :\n\t{bot.content}")    # chatmodel answer
        print()
    except Exception as e:    # Handle errors
        print("error :",e)