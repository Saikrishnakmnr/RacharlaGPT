import os
import streamlit as st
from llm_chain import get_groq_response

# Streamlit Page Setup
st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

# API Key Validation from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

# Sidebar Configuration
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

# New Chat button to reset conversation state
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Initialize Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages on screen reload
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
if prompt := st.chat_input("Type your message..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response using llm_chain.py
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_groq_response(st.session_state.messages)
            st.markdown(response)

    # Append assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
