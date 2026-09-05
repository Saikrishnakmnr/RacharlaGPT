import os
import uuid
from datetime import datetime

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# RACHARLAGPT — ChatGPT-style Groq assistant
# ============================================================

st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# App configuration
# -----------------------------
PRIMARY_MODEL = "openai/gpt-oss-20b"
BACKUP_MODEL = "llama-3.1-8b-instant"

DEFAULT_SYSTEM_PROMPT = """You are RacharlaGPT, a helpful, intelligent, friendly and concise AI assistant.
Give accurate, clear and useful answers.
Use headings, bullet points, numbered steps and tables when they improve readability.
If the user asks for code, provide complete working code and explain important changes briefly.
Do not invent facts. If you are uncertain, say so.
Be practical and solution-oriented."""

# Keep the complete conversation visible in the UI, but only send recent
# messages to the model. This helps control token usage.
MAX_CONTEXT_MESSAGES = 14


# ============================================================
# Secrets / API key
# ============================================================
def get_api_key():
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None

    return key


api_key = get_api_key()

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.info(
        "For Streamlit Cloud: open your app → Settings → Secrets and add "
        'GROQ_API_KEY = "your_groq_api_key_here"'
    )
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #ffffff;
        }

        /* ---------- Header ---------- */
        .app-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 6px 0 4px 0;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            background: linear-gradient(135deg, #ff4b4b, #ff9f1c);
            box-shadow: 0 8px 24px rgba(255, 75, 75, 0.20);
        }

        .brand-name {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -1px;
            color: #172033;
        }

        .brand-subtitle {
            color: #6b7280;
            margin: 0 0 24px 55px;
            font-size: 14px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #f7f8fc;
            border-right: 1px solid #e7e9ef;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 22px;
            font-weight: 800;
            color: #172033;
            margin-bottom: 14px;
        }

        .sidebar-badge {
            font-size: 20px;
        }

        .history-label {
            margin-top: 22px;
            margin-bottom: 8px;
            color: #8a8f9d;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .8px;
        }

        .empty-history {
            color: #8a8f9d;
            font-size: 13px;
            padding: 12px 6px;
        }

        /* Make chat-history buttons look like clean list items. */
        div[data-testid="stSidebar"] button {
            border-radius: 11px !important;
        }

        /* ---------- Chat area ---------- */
        .welcome-card {
            max-width: 760px;
            margin: 10vh auto 3vh auto;
            text-align: center;
            padding: 36px 24px;
        }

        .welcome-logo {
            font-size: 48px;
            margin-bottom: 8px;
        }

        .welcome-title {
            font-size: 34px;
            font-weight: 800;
            color: #172033;
            letter-spacing: -1px;
        }

        .welcome-text {
            color: #6b7280;
            font-size: 15px;
            line-height: 1.6;
        }

        .suggestion {
            border: 1px solid #e7e9ef;
            border-radius: 14px;
            padding: 14px 16px;
            background: #fbfcff;
            margin: 8px 0;
            text-align: left;
            color: #3f4654;
        }

        .chat-footer-note {
            text-align: center;
            color: #9aa0ad;
            font-size: 11px;
            margin-top: 12px;
        }

        /* ---------- Status card ---------- */
        .status-card {
            border: 1px solid #e7e9ef;
            border-radius: 13px;
            padding: 10px 12px;
            background: white;
            font-size: 12px;
            color: #596170;
        }

        /* ---------- Mobile ---------- */
        @media (max-width: 700px) {
            .brand-name {
                font-size: 25px;
            }

            .welcome-title {
                font-size: 27px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================
def new_chat_data():
    return {
        "title": "New Chat",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated": datetime.now().timestamp(),
        "messages": [],
    }


if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {first_id: new_chat_data()}
    st.session_state.current_chat_id = first_id

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = next(iter(st.session_state.chats))

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "search_history" not in st.session_state:
    st.session_state.search_history = ""

if "busy" not in st.session_state:
    st.session_state.busy = False


# ============================================================
# Chat helpers
# ============================================================
def create_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = new_chat_data()
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]

    if not st.session_state.chats:
        create_chat()
        return

    if st.session_state.current_chat_id == chat_id:
        # Open the most recently updated remaining chat.
        remaining = sorted(
            st.session_state.chats.items(),
            key=lambda item: item[1]["updated"],
            reverse=True,
        )
        st.session_state.current_chat_id = remaining[0][0]


def delete_all_chats():
    first_id = str(uuid.uuid4())
    st.session_state.chats = {first_id: new_chat_data()}
    st.session_state.current_chat_id = first_id


def current_chat():
    chat_id = st.session_state.current_chat_id

    if chat_id not in st.session_state.chats:
        create_chat()

    return st.session_state.chats[st.session_state.current_chat_id]


def make_title(text):
    cleaned = " ".join(text.strip().split())

    if not cleaned:
        return "New Chat"

    # Keep sidebar titles compact.
    if len(cleaned) <= 34:
        return cleaned

    return cleaned[:34].rstrip() + "…"


def chat_matches(chat, query):
    if not query:
        return True

    q = query.lower()

    if q in chat["title"].lower():
        return True

    for message in chat["messages"]:
        if q in message["content"].lower():
            return True

    return False


def build_model_messages(chat):
    recent = chat["messages"][-MAX_CONTEXT_MESSAGES:]

    messages = [SystemMessage(content=st.session_state.system_prompt)]

    for message in recent:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))

    return messages


# ============================================================
# Model setup
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models(key):
    primary = ChatGroq(
        model=PRIMARY_MODEL,
        temperature=0.7,
        api_key=key,
    )

    backup = ChatGroq(
        model=BACKUP_MODEL,
        temperature=0.7,
        api_key=key,
    )

    return primary, backup


primary_llm, backup_llm = load_models(api_key)


def ask_model(chat):
    messages = build_model_messages(chat)

    # Primary model first.
    try:
        return primary_llm.invoke(messages), "primary"

    except RateLimitError:
        # If the primary is temporarily rate-limited, try the lighter
        # backup model.
        try:
            return backup_llm.invoke(messages), "backup"

        except RateLimitError as backup_error:
            raise backup_error


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-badge">⚡</span>
            <span>RacharlaGPT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New Chat",
        use_container_width=True,
        type="primary",
        key="new_chat_btn",
    ):
        create_chat()
        st.rerun()

    st.markdown('<div class="history-label">YOUR CHATS</div>', unsafe_allow_html=True)

    search = st.text_input(
        "Search chats",
        value=st.session_state.search_history,
        placeholder="🔎 Search your chats...",
        label_visibility="collapsed",
        key="chat_search_box",
    )
    st.session_state.search_history = search

    chats_sorted = sorted(
        st.session_state.chats.items(),
        key=lambda item: item[1]["updated"],
        reverse=True,
    )

    visible_chats = [
        (chat_id, chat)
        for chat_id, chat in chats_sorted
        if chat_matches(chat, search)
    ]

    if not visible_chats:
        st.markdown(
            '<div class="empty-history">No chats match your search.</div>',
            unsafe_allow_html=True,
        )

    for chat_id, chat in visible_chats:
        col_chat, col_delete = st.columns([0.80, 0.20], gap="small")

        is_current = chat_id == st.session_state.current_chat_id

        with col_chat:
            if st.button(
                f"💬  {chat['title']}",
                key=f"open_{chat_id}",
                use_container_width=True,
                type="secondary" if not is_current else "primary",
                help="Open this conversation",
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        with col_delete:
            if st.button(
                "🗑️",
                key=f"delete_{chat_id}",
                use_container_width=True,
                help=f"Delete '{chat['title']}'",
            ):
                delete_chat(chat_id)
                st.rerun()

    st.divider()

    if st.button(
        "🗑️  Delete All Chats",
        use_container_width=True,
        key="delete_all_btn",
    ):
        delete_all_chats()
        st.rerun()

    st.divider()

    st.markdown("### ⚙️ Configuration")

    st.session_state.system_prompt = st.text_area(
        "AI System Prompt",
        value=st.session_state.system_prompt,
        height=155,
        help="Controls the assistant's personality and response style.",
    )

    st.markdown(
        f"""
        <div class="status-card">
            <b>Model</b><br>
            {PRIMARY_MODEL}<br><br>
            <b>Fallback</b><br>
            {BACKUP_MODEL}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Your visible chat history stays in this browser session. "
        "For permanent multi-device history, connect a database such as Supabase later."
    )


# ============================================================
# Main header
# ============================================================
st.markdown(
    """
    <div class="app-brand">
        <div class="brand-icon">⚡</div>
        <div class="brand-name">RacharlaGPT</div>
    </div>
    <div class="brand-subtitle">Fast, intelligent AI conversations powered by Groq</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Current conversation
# ============================================================
chat = current_chat()

if not chat["messages"]:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-logo">⚡</div>
            <div class="welcome-title">Welcome to RacharlaGPT</div>
            <div class="welcome-text">
                Ask questions, write code, brainstorm ideas, learn something new,
                or simply have a conversation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestion_cols = st.columns(3)

    suggestions = [
        ("💡 Learn", "Explain a difficult topic simply"),
        ("💻 Code", "Build a Streamlit application"),
        ("🚀 Create", "Give me a creative project idea"),
    ]

    for col, (heading, prompt_text) in zip(suggestion_cols, suggestions):
        with col:
            st.markdown(
                f"""
                <div class="suggestion">
                    <b>{heading}</b><br>
                    <span>{prompt_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Render complete visible history.
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat input
# ============================================================
user_input = st.chat_input(
    "Message RacharlaGPT...",
    disabled=st.session_state.busy,
)

if user_input:
    user_input = user_input.strip()

    if user_input:
        # Save and display the user's message immediately.
        chat["messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        chat["updated"] = datetime.now().timestamp()

        # Automatically name a new conversation from its first question.
        if chat["title"] == "New Chat":
            chat["title"] = make_title(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("RacharlaGPT is thinking…"):
                    response, model_used = ask_model(chat)

                answer = response.content

                if not isinstance(answer, str):
                    answer = str(answer)

                # Save assistant response.
                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                chat["updated"] = datetime.now().timestamp()

                if model_used == "backup":
                    st.caption(
                        f"Primary model was temporarily rate-limited. "
                        f"Answered using `{BACKUP_MODEL}`."
                    )

                st.markdown(answer)

            except RateLimitError:
                # Friendly handling instead of exposing a traceback.
                # This is especially useful when Groq daily token limits
                # have been exhausted.
                friendly = (
                    "⚠️ **Groq rate limit reached.**\n\n"
                    "The available Groq quota for the selected models is "
                    "temporarily exhausted. This is an API quota issue, not "
                    "a problem with your question.\n\n"
                    "Please wait for the quota window to reset, then try again."
                )

                st.markdown(friendly)

                # Remove the unsent user message so the UI does not pretend
                # the question received a real answer.
                if chat["messages"] and chat["messages"][-1]["role"] == "user":
                    chat["messages"].pop()

            except Exception as exc:
                st.error(
                    "Something went wrong while contacting Groq. "
                    "Please try again."
                )

                # Keep the actual error out of the normal UI, but leave
                # enough information for debugging in the server log.
                print(f"RacharlaGPT error: {type(exc).__name__}: {exc}")

                if chat["messages"] and chat["messages"][-1]["role"] == "user":
                    chat["messages"].pop()


st.markdown(
    '<div class="chat-footer-note">RacharlaGPT • Powered by Groq • AI can make mistakes, so verify important information.</div>',
    unsafe_allow_html=True,
)
