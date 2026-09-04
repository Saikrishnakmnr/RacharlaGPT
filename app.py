import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="⚡ Groq AI Assistant", page_icon="⚡", layout="wide")
st.title("⚡ Groq AI Assistant")

# Fetch key safely from secrets or environment
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY not found! Please set it in .streamlit/secrets.toml")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# Sidebar setup
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):", 
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Initialize models with fallback protection
primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
llm = primary_llm.with_fallbacks([backup_llm])

# Chat History setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Chat Input
if user_input := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.write(user_input)
    
    st.session_state.messages.append(HumanMessage(content=user_input))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "history": st.session_state.messages[:-1],
                "input": user_input
            })
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))
