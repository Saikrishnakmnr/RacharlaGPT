import os
import streamlit as st
from llm_chain import get_groq_response

# 1. Page Config
st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

# 2. Secret Check
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

# 3. Sidebar UI
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

# 4. New Chat Reset Button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 5. Session State Memory Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Render visible chat UI history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Prompt Execution Engine
if prompt := st.chat_input("Type your message..."):
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # CRITICAL FIX: Slice session memory directly HERE in app.py
    # Send ONLY the last 2 items (current user prompt + 1 prior assistant response)
    payload_to_send = st.session_state.messages[-2:]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_groq_response(payload_to_send, system_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
