# practice 1

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
# from langchain_core.messages import HumanMessage,AIMessage
# from dotenv import load_dotenv


# load_dotenv()

# model = ChatGroq(model="llama-3.1-8b-instant")

# template = ChatPromptTemplate([
#     ('system',"\nYou are a helpful AI assistant.\n\nRules:\n- Give short and meaningful answers.\n- Use simple English.\n- Answer directly.\n- Keep responses within 2-4 sentences.\n- Avoid unnecessary details.\n\nHistory Usage Rules:\n- Chat history is only for context.\n- Use chat history ONLY if the user's current question clearly refers to previous messages.\n- Examples of history-related questions:\n  - \"What did I ask before?\"\n  - \"Continue that topic.\"\n  - \"Explain it again.\"\n  - \"What was my previous project?\"\n  - Questions containing words like: previous, before, earlier, continue, that, it, this topic.\n\n- If the current question is a completely new topic, IGNORE the chat history entirely.\n- Never mention chat history unless the user explicitly asks about previous conversation.\n- Do not say \"According to your previous chat\" or similar phrases unless the user asks about previous messages.\n- For new questions, answer only from the current question."),
#     MessagesPlaceholder(variable_name='history'),
#     ('human','{input}')
# ])

# temp = load_prompt("template1.json")
# temmm = temp.template

# pre_chat = []
# with open('history.txt','r') as f:
#     pre_chat.extend(f.readline)

# doc = [] # runing chat 

# while True:
#     user = input("USER: ")
#     doc.append(HumanMessage(content=user))
#     if user.upper() == "EXIT":
#         print("chat end...")
#         with open("history.txt", "w") as f:
#             for msg in doc:
#                 f.write(
#                     f"{msg.__class__.__name__}: "
#                     f"{msg.content}\n"
#                 )
#         break
#     try:
#         tem = template.format(input=user,history=pre_chat+doc)
#         result = model.invoke(tem)
#         print("BOT: ",result.content)
#         doc.append(AIMessage(content=result.content))
#     except Exception as e:
#         print("ERROR:",e)




# practice 2

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import (
#     HumanMessage,
#     AIMessage,
#     SystemMessage
# )
# from dotenv import load_dotenv

# # =========================
# # Load Environment
# # =========================

# load_dotenv()

# # =========================
# # Model
# # =========================

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=1
# )

# # =========================
# # Load Previous History
# # =========================

# pre_chat = []

# try:
#     with open("history.txt", "r") as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("HumanMessage:"):
#                 pre_chat.append(
#                     HumanMessage(
#                         content=line.replace(
#                             "HumanMessage:",
#                             ""
#                         ).strip()
#                     )
#                 )
#             elif line.startswith("AIMessage:"):
#                 pre_chat.append(
#                     AIMessage(
#                         content=line.replace(
#                             "AIMessage:",
#                             ""
#                         ).strip()
#                     )
#                 )

# except FileNotFoundError:

#     print("No previous history found.")

# # =========================
# # Prompt
# # =========================

# template = ChatPromptTemplate.from_messages(

#     [

#         (
#             "system",
#             """
# You are a helpful AI assistant.

# Rules:
# - Give short and meaningful answers.
# - Use simple English.
# - Answer directly.
# - Keep responses within 2-4 sentences.
# - Avoid unnecessary details.

# History Usage Rules:
# - Chat history is only for context.
# - Use chat history ONLY if the current question depends on previous messages.
# - If the question is complete by itself, answer directly.
# - Never mention chat history unless the user asks about it.
# - Resolve pronouns like:
#   he, she, him, her, they, it
#   using previous conversation when needed.
# """
#         ),

#         MessagesPlaceholder(variable_name="history"),

#         (
#             "human",
#             "{input}"
#         )

#     ]

# )

# # =========================
# # Current Session History
# # =========================

# doc = []

# print("\nAsk Questions!!")

# # =========================
# # Chat Loop
# # =========================

# while True:

#     user = input("⦾ User: ")
#     if user.lower() == "exit":
#         print("chatbot ended...")
#         with open("history.txt", "w") as f:
#             all_history = pre_chat + doc
#             for msg in all_history:
#                 f.write(
#                     f"{msg.__class__.__name__}: "
#                     f"{msg.content}\n"
#                 )

#         print("History saved.")
#         break
#     try:
#         # Build history
#         full_history = pre_chat + doc
#         # Create messages
#         messages = template.format_messages(
#             history=full_history,
#             input=user
#         )
#         # Get response
#         result = model.invoke(messages)
#         print(f"⦿ Bot :\n\t{result.content}\n")
#         # Save current session history
#         doc.append(
#             HumanMessage(content=user)
#         )
#         doc.append(
#             AIMessage(content=result.content)
#         )
#     except Exception as e:
#         print("ERROR:", e)

# practice 3

from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 1.5
)

prompt1 = ChatPromptTemplate([
    ("system",  "you are a python teacher with 15+ years of experience"),
    ("human" , "give me detaild notes about topic {topic}")
])

parser = StrOutputParser()

prompt2 = ChatPromptTemplate([
    ("system",  "you are a python teacher with 15+ years of experience"),
    ("human" , "give me 5 mcq question {notes}")
])

chain = prompt1 | model | parser 

chains = RunnableParallel({
    "notes" : RunnablePassthrough(),
    "mcq" : RunnableSequence(prompt2,model,parser) # prompt2 | model | parser
})

final_chain = chain | chains


user = input("ask...")

result = final_chain.invoke({"topic" : "OOPs"})

print(result["mcq"])