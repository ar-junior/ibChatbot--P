# import AI model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv

# Get API key
load_dotenv()

# Model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1.2
)

template = ChatPromptTemplate([
    ("system",
    """
You are a helpful AI assistant.

Primary Objective:
Understand the user's current question first.
Then decide whether chat history is needed.

Rules:
- Give short, meaningful, and direct answers.
- Use simple English.
- Keep responses within 2-4 sentences.
- Avoid unnecessary details.

History Rules:

Step 1:
Analyze the current question.

Step 2:
Check whether the question depends on previous conversation.

Use history ONLY when:
- The user uses pronouns such as:
  he, she, him, her, they, them, it
- The user asks follow-up questions such as:
  "and him?"
  "and virat?"
  "what about him?"
  "is he married?"
  "continue"
  "tell me more"
  "what was that?"

- The current question cannot be understood correctly without previous messages.

Ignore history when:
- The current question is complete by itself.
- The question contains enough information to answer directly.
- The user starts a new topic.

Important:
- First try to answer from the current question alone.
- Use history only if the question becomes ambiguous without history.
- Never say "there is no information in history" or "history does not contain the answer".
- History is only for resolving references and follow-up questions.
- If the answer is common knowledge, answer directly.
"""),
    MessagesPlaceholder(variable_name='history'),
    ("human","{input}")

])
pre_chat = [] # message object

try:    
    with open ("previous_chat.txt",'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("HumanMessage:"):
                pre_chat.append(
                    HumanMessage(
                        content=line.replace(
                            "HumanMessage:",""
                        ).strip()
                    )
                )
            elif line.startswith("AIMessage:"):
                pre_chat.append(
                    AIMessage(
                        content=line.replace(
                            "AIMessage:",""
                        ).strip()
                    )
                )
except FileNotFoundError:
    print("file not found")

doc = [] # message object

# template = load_prompt("template1.json")   # get prompttemplate from template.json file

print("\nAsk Questions...")    # chatbot hader

while True:   # input/output loop
    
    user = input("⦾ User: ")    # user input
    doc.append(HumanMessage(content=user))

    if user.lower()=="exit":    # exit from chat
        print("chatbot ended...")
        with open('previous_chat.txt','w') as f:
            chat = pre_chat + doc
            for msg in chat:
                f.write(
                    f"{msg.__class__.__name__}:"
                    f"{msg.content}\n"
                )

        break
    try:
        tmp_ans = template.format_messages(input=user,history =pre_chat+doc)   # send input to template
        bot = model.invoke(tmp_ans)
        print(f"⦿ Bot :\n\t{bot.content}")    # chatmodel answer
        doc.append(AIMessage(content=bot.content))
        # doc.append({"User":user,"bot":bot.content})   # chat history
        print()
    except Exception as e:    # Handle errors
        print("error :",e)


