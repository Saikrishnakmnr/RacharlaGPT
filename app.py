import os
import streamlit as st
from llm_chain import get_groq_response

# 1. Page Config
st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

# 2. Verify Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

# 3. Persona Setting
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

# 4. Initialize Local Chat State for UI display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Render visible screen bubbles
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Prompt Input
if prompt := st.chat_input("Type your message..."):
    # Append and show user query
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant output
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_groq_response(prompt, system_prompt)
                st.markdown(response)
                # Store output string
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
