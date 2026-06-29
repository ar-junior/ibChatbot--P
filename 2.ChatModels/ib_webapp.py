# from langchain_groq import ChatGroq
# from langchain_core.prompts import load_prompt
# from langchain_core.messages import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )
# from dotenv import load_dotenv
# import streamlit as st

# # Load API key
# load_dotenv()

# # Model
# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=1.2
# )

# template = load_prompt("template1.json")

# # Page config
# st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

# st.title("🤖 Groq AI Chatbot")

# # Session state
# if "history" not in st.session_state:
#     st.session_state.history = [
#         SystemMessage(content="You are a helpful AI assistant.")
#     ]

# # Display old chats
# for msg in st.session_state.history:
#     if isinstance(msg, HumanMessage):
#         with st.chat_message("user"):
#             st.write(msg.content)

#     elif isinstance(msg, AIMessage):
#         with st.chat_message("assistant"):
#             st.write(msg.content)

# # User input
# user = st.chat_input("Ask something...")

# if user:

#     # Show user message
#     with st.chat_message("user"):
#         st.write(user)

#     st.session_state.history.append(
#         HumanMessage(content=user)
#     )

#     try:
#         prompt = template.format(
#             input=user,
#             history=st.session_state.history
#         )

#         response = model.invoke(prompt)

#         with st.chat_message("assistant"):
#             st.write(response.content)

#         st.session_state.history.append(
#             AIMessage(content=response.content)
#         )

#     except Exception as e:
#         st.error(f"Error: {e}")



from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

# =========================
# Load Environment
# =========================

load_dotenv()

# =========================
# Model
# =========================

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1.2
)

# =========================
# Prompt
# =========================

template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Rules:
- Give short, meaningful answers.
- Use simple English.
- Keep responses within 2-4 sentences.
- Use history only when needed.
"""
    ),

    MessagesPlaceholder("history"),

    ("human", "{input}")
])

# =========================
# Load Previous History
# =========================

pre_chat = []

try:

    with open("previous_chat.txt", "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("HumanMessage:"):

                pre_chat.append(
                    HumanMessage(
                        content=line.replace(
                            "HumanMessage:",
                            ""
                        ).strip()
                    )
                )

            elif line.startswith("AIMessage:"):

                pre_chat.append(
                    AIMessage(
                        content=line.replace(
                            "AIMessage:",
                            ""
                        ).strip()
                    )
                )

except FileNotFoundError:
    pass

# =========================
# Session State
# =========================

if "history" not in st.session_state:
    st.session_state.history = pre_chat.copy()

# =========================
# UI
# =========================

st.title("🤖 LangChain Chatbot")

# Show old messages

for msg in st.session_state.history:

    if isinstance(msg, HumanMessage):

        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):

        with st.chat_message("assistant"):
            st.write(msg.content)

# =========================
# User Input
# =========================

user = st.chat_input("Ask anything...")

if user:

    # show user message

    with st.chat_message("user"):
        st.write(user)

    # build prompt

    messages = template.format_messages(
        input=user,
        history=st.session_state.history
    )

    # model response

    response = model.invoke(messages)

    # show bot response

    with st.chat_message("assistant"):
        st.write(response.content)

    # save in memory

    st.session_state.history.append(
        HumanMessage(content=user)
    )

    st.session_state.history.append(
        AIMessage(content=response.content)
    )

    # save to file

    with open("previous_chat.txt", "w") as f:

        for msg in st.session_state.history:

            f.write(
                f"{msg.__class__.__name__}:"
                f"{msg.content}\n"
            )