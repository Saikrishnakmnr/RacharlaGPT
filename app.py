```python
import os
import uuid
from datetime import datetime

import streamlit as st

from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="⚡ Groq AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 320px;
    }

    /* Chat title */
    .app-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Chat history buttons */
    .chat-history-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #888;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Welcome screen */
    .welcome-box {
        text-align: center;
        padding: 4rem 1rem 2rem 1rem;
    }

    .welcome-box h2 {
        font-size: 2rem;
    }

    .welcome-box p {
        color: #888;
        font-size: 1rem;
    }

    /* Small status text */
    .small-text {
        color: #888;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "🔐 GROQ_API_KEY was not found.\n\n"
        "Add it to your Streamlit secrets:\n\n"
        "`GROQ_API_KEY = \"your_api_key_here\"`"
    )
    st.stop()


# ============================================================
# CONSTANTS
# ============================================================

# Number of recent messages sent to the model.
#
# IMPORTANT:
# We keep ALL messages in the UI/chat history,
# but only send recent messages to Groq.
#
# This prevents long conversations from consuming
# excessive tokens.
MAX_CONTEXT_MESSAGES = 12

PRIMARY_MODEL = "openai/gpt-oss-20b"
BACKUP_MODEL = "qwen/qwen3.8-27b"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, friendly and concise AI assistant. "
    "Give accurate, clear and useful answers. "
    "Use headings, bullet points and examples when they improve readability. "
    "If you are unsure about something, say so instead of inventing facts."
)


# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT


# ============================================================
# CHAT MANAGEMENT FUNCTIONS
# ============================================================

def create_chat():
    """
    Create a new empty conversation.
    """
    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }

    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    """
    Delete one conversation.
    """

    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]

    # If the deleted chat was active,
    # automatically open another chat.
    if st.session_state.current_chat_id == chat_id:

        if st.session_state.chats:
            # Open the newest remaining chat
            remaining = sorted(
                st.session_state.chats.items(),
                key=lambda item: item[1]["created_at"],
                reverse=True,
            )

            st.session_state.current_chat_id = remaining[0][0]

        else:
            create_chat()


def delete_all_chats():
    """
    Delete every conversation and start a fresh chat.
    """
    st.session_state.chats = {}
    create_chat()


def get_current_chat():
    """
    Return the currently selected chat.
    """

    chat_id = st.session_state.current_chat_id

    if chat_id is None:
        create_chat()
        chat_id = st.session_state.current_chat_id

    if chat_id not in st.session_state.chats:
        create_chat()
        chat_id = st.session_state.current_chat_id

    return st.session_state.chats[chat_id]


def generate_chat_title(user_message):
    """
    Generate a simple title from the first user question.
    """

    title = " ".join(user_message.strip().split())

    if not title:
        return "New Chat"

    # Keep sidebar titles short
    if len(title) > 42:
        title = title[:42].rstrip() + "..."

    return title


# ============================================================
# INITIAL CHAT
# ============================================================

if not st.session_state.chats:
    create_chat()


current_chat = get_current_chat()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚡ Groq AI")

    # New Chat button
    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        create_chat()
        st.rerun()

    st.markdown(
        '<div class="chat-history-title">YOUR CHATS</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if st.session_state.chats:

        # Newest conversations first
        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda item: item[1]["created_at"],
            reverse=True,
        )

        for chat_id, chat in sorted_chats:

            col1, col2 = st.columns([5, 1])

            with col1:

                # Highlight currently selected chat
                if chat_id == st.session_state.current_chat_id:
                    button_label = f"💬 {chat['title']}"
                else:
                    button_label = chat["title"]

                if st.button(
                    button_label,
                    key=f"open_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}",
                    help="Delete this chat",
                ):
                    delete_chat(chat_id)
                    st.rerun()

    else:
        st.caption("No conversations yet.")

    st.divider()

    # --------------------------------------------------------
    # DELETE ALL
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete All Chats",
        use_container_width=True,
    ):
        delete_all_chats()
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    st.markdown("### ⚙️ Configuration")

    st.session_state.system_prompt = st.text_area(
        "AI System Prompt",
        value=st.session_state.system_prompt,
        height=150,
        help="Controls the personality and behavior of the AI.",
    )

    st.caption(f"Primary model: `{PRIMARY_MODEL}`")
    st.caption(f"Fallback model: `{BACKUP_MODEL}`")

    st.divider()

    st.markdown(
        """
        <div class="small-text">
        ⚡ Powered by Groq<br>
        💬 Chat history is maintained for this session<br>
        🔒 API key is stored in Streamlit Secrets
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="app-title">⚡ Groq AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">Fast, intelligent AI conversations</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================

current_chat = get_current_chat()

for message in current_chat["messages"]:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not current_chat["messages"]:

    st.markdown(
        """
        <div class="welcome-box">
            <h2>How can I help you today?</h2>
            <p>
                Ask me anything about AI, LLMs, Python,
                programming, mathematics, science, or general topics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CREATE GROQ MODELS
# ============================================================

@st.cache_resource
def get_models(api_key):

    primary = ChatGroq(
        model=PRIMARY_MODEL,
        temperature=0.7,
        max_tokens=2048,
        groq_api_key=api_key,
    )

    backup = ChatGroq(
        model=BACKUP_MODEL,
        temperature=0.7,
        max_tokens=2048,
        groq_api_key=api_key,
    )

    return primary, backup


primary_llm, backup_llm = get_models(api_key)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Message Groq AI..."
)


if user_input:

    user_input = user_input.strip()

    if not user_input:
        st.stop()

    # --------------------------------------------------------
    # CURRENT CHAT
    # --------------------------------------------------------

    current_chat = get_current_chat()

    # --------------------------------------------------------
    # FIRST QUESTION = CHAT TITLE
    # --------------------------------------------------------

    if len(current_chat["messages"]) == 0:
        current_chat["title"] = generate_chat_title(user_input)

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # PREPARE HISTORY FOR MODEL
    # --------------------------------------------------------
    #
    # The UI keeps the complete conversation.
    #
    # But Groq receives only the last N messages.
    #
    # This is important for reducing token usage.
    # --------------------------------------------------------

    all_messages = current_chat["messages"]

    recent_messages = all_messages[-MAX_CONTEXT_MESSAGES:]

    model_history = []

    # Don't send the current question twice.
    # The current question is passed separately as {input}.
    previous_messages = recent_messages[:-1]

    for message in previous_messages:

        if message["role"] == "user":
            model_history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":
            model_history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # --------------------------------------------------------
    # BUILD CHAIN
    # --------------------------------------------------------

    primary_chain = prompt | primary_llm
    backup_chain = prompt | backup_llm

    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    response = None

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # ==================================================
            # TRY PRIMARY MODEL
            # ==================================================

            try:

                response = primary_chain.invoke(
                    {
                        "system_prompt": st.session_state.system_prompt,
                        "history": model_history,
                        "input": user_input,
                    }
                )

            # ==================================================
            # RATE LIMIT
            # ==================================================

            except RateLimitError:

                st.warning(
                    "⚠️ The primary Groq model has reached "
                    "its current rate limit. Trying the backup model..."
                )

                # Try backup model
                try:

                    response = backup_chain.invoke(
                        {
                            "system_prompt": st.session_state.system_prompt,
                            "history": model_history,
                            "input": user_input,
                        }
                    )

                except RateLimitError:

                    st.error(
                        "⚠️ Both Groq models are currently rate-limited.\n\n"
                        "Please wait a few minutes and try again."
                    )

                except Exception as backup_error:

                    st.error(
                        f"⚠️ Backup model error: {backup_error}"
                    )

            # ==================================================
            # OTHER PRIMARY ERROR
            # ==================================================

            except Exception as primary_error:

                st.warning(
                    "⚠️ The primary model could not answer. "
                    "Trying the backup model..."
                )

                try:

                    response = backup_chain.invoke(
                        {
                            "system_prompt": st.session_state.system_prompt,
                            "history": model_history,
                            "input": user_input,
                        }
                    )

                except RateLimitError:

                    st.error(
                        "⚠️ The backup model is also rate-limited. "
                        "Please try again later."
                    )

                except Exception as backup_error:

                    st.error(
                        "⚠️ Both models failed.\n\n"
                        f"Primary error: {primary_error}\n\n"
                        f"Backup error: {backup_error}"
                    )

            # ==================================================
            # SAVE AI RESPONSE
            # ==================================================

            if response is not None:

                answer = response.content

                st.markdown(answer)

                current_chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

    # Refresh sidebar so the new chat title appears immediately
    st.rerun()
```
