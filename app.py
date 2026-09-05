import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from supabase import create_client


# =========================================================
# RacharlaGPT
# Stable version
# - Supabase login/signup
# - Permanent chat history
# - New Chat
# - Search chats
# - Rename chat
# - Delete chat
# - Delete all chats
# - Download chat
# - AI settings
# - Groq fallback
# - NO COOKIE PACKAGE
# - NO COOKIE_PASSWORD
# =========================================================

APP_NAME = "RacharlaGPT"

PRIMARY_MODEL = "openai/gpt-oss-20b"
BACKUP_MODEL = "llama-3.1-8b-instant"

MAX_CONTEXT_MESSAGES = 14


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1100px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .rach-header {
        text-align: center;
        padding: 10px 0 18px 0;
    }

    .rach-logo {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .rach-title {
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .rach-subtitle {
        color: #777;
        font-size: 14px;
    }

    /* Auth card */
    .auth-box {
        max-width: 520px;
        margin: 45px auto 0 auto;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        min-width: 290px;
    }

    /* Mobile */
    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-top: 0.6rem;
        }

        .rach-logo {
            font-size: 34px;
        }

        .rach-title {
            font-size: 24px;
        }

        section[data-testid="stSidebar"] {
            min-width: 250px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SECRETS
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

except Exception:
    st.error(
        "Missing Streamlit Secrets. Please add GROQ_API_KEY, "
        "SUPABASE_URL and SUPABASE_KEY."
    )
    st.stop()


# =========================================================
# SUPABASE
# =========================================================

@st.cache_resource
def get_supabase():
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )


supabase = get_supabase()


# =========================================================
# SESSION STATE
# =========================================================

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Auto (recommended)"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are RacharlaGPT, a helpful, intelligent and friendly AI assistant. "
        "Answer clearly and accurately. Keep responses natural and useful."
    )


# =========================================================
# AUTH HELPERS
# =========================================================

def show_auth():
    """Show login/signup screen."""

    st.markdown(
        """
        <div class="auth-box">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="rach-header">'
        '<div class="rach-logo">⚡</div>'
        '<div class="rach-title">Welcome to RacharlaGPT</div>'
        '<div class="rach-subtitle">Your personal AI assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(
        ["🔐 Sign In", "📝 Create Account"]
    )

    with login_tab:

        email = st.text_input(
            "Email",
            placeholder="you@example.com",
            key="login_email",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Sign In",
            type="primary",
            use_container_width=True,
        ):

            email = email.strip()

            if not email or not password:
                st.warning("Please enter your email and password.")
            else:

                try:
                    response = supabase.auth.sign_in_with_password(
                        {
                            "email": email,
                            "password": password,
                        }
                    )

                    if response.user:
                        st.session_state.auth_user = response.user
                        st.session_state.chats = {}
                        st.session_state.current_chat_id = None
                        st.rerun()

                except Exception as e:
                    st.error(
                        "Sign in failed. Please check your email and password."
                    )

    with signup_tab:

        signup_email = st.text_input(
            "Email",
            placeholder="you@example.com",
            key="signup_email",
        )

        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
        )

        signup_password2 = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_password2",
        )

        if st.button(
            "Create Account",
            type="primary",
            use_container_width=True,
        ):

            signup_email = signup_email.strip()

            if not signup_email or not signup_password:
                st.warning("Please enter an email and password.")

            elif signup_password != signup_password2:
                st.warning("Passwords do not match.")

            elif len(signup_password) < 6:
                st.warning("Password must be at least 6 characters.")

            else:

                try:
                    response = supabase.auth.sign_up(
                        {
                            "email": signup_email,
                            "password": signup_password,
                        }
                    )

                    if response.user:

                        if response.session:
                            st.session_state.auth_user = response.user
                            st.session_state.chats = {}
                            st.session_state.current_chat_id = None
                            st.success("Account created successfully.")
                            st.rerun()

                        else:
                            st.success(
                                "Account created. Please check your email "
                                "to confirm your account, then sign in."
                            )

                except Exception:
                    st.error(
                        "Could not create the account. "
                        "The email may already be registered."
                    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# AUTH GATE
# =========================================================

if st.session_state.auth_user is None:
    show_auth()
    st.stop()


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def load_chats():
    """Load all chats belonging to current user."""

    user_id = st.session_state.auth_user.id

    try:
        response = (
            supabase
            .table("chats")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )

        chats = {}

        for row in response.data or []:
            chats[row["id"]] = {
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row.get("title") or "New Chat",
                "messages": row.get("messages") or [],
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }

        st.session_state.chats = chats

    except Exception as e:
        st.error("Could not load your chat history.")


def save_chat(chat):
    """Save existing chat to Supabase."""

    now = datetime.now(timezone.utc).isoformat()

    data = {
        "id": chat["id"],
        "user_id": st.session_state.auth_user.id,
        "title": chat["title"],
        "messages": chat["messages"],
        "updated_at": now,
    }

    try:
        supabase.table("chats").upsert(data).execute()

        chat["updated_at"] = now
        st.session_state.chats[chat["id"]] = chat

    except Exception:
        st.error("Could not save this chat.")


def create_chat():
    """Create a new empty chat."""

    chat_id = str(uuid.uuid4())

    chat = {
        "id": chat_id,
        "user_id": st.session_state.auth_user.id,
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        supabase.table("chats").insert(chat).execute()

        st.session_state.chats[chat_id] = chat
        st.session_state.current_chat_id = chat_id

    except Exception:
        st.error("Could not create a new chat.")


def delete_chat(chat_id):
    """Delete one chat."""

    try:
        (
            supabase
            .table("chats")
            .delete()
            .eq("id", chat_id)
            .eq("user_id", st.session_state.auth_user.id)
            .execute()
        )

        st.session_state.chats.pop(chat_id, None)

        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None

    except Exception:
        st.error("Could not delete the chat.")


def delete_all_chats():
    """Delete all chats belonging to current user."""

    try:
        (
            supabase
            .table("chats")
            .delete()
            .eq("user_id", st.session_state.auth_user.id)
            .execute()
        )

        st.session_state.chats = {}
        st.session_state.current_chat_id = None

    except Exception:
        st.error("Could not delete your chats.")


# =========================================================
# LOAD HISTORY
# =========================================================

if not st.session_state.chats:
    load_chats()

if (
    st.session_state.current_chat_id is None
    and st.session_state.chats
):
    st.session_state.current_chat_id = next(
        iter(st.session_state.chats)
    )


# =========================================================
# CHAT FUNCTIONS
# =========================================================

def generate_title(text):
    """Generate a simple title without another AI request."""

    text = " ".join(text.strip().split())

    if not text:
        return "New Chat"

    if len(text) <= 45:
        return text

    return text[:45].rstrip() + "..."


def get_model_name():
    selected = st.session_state.selected_model

    if selected == PRIMARY_MODEL:
        return PRIMARY_MODEL

    if selected == BACKUP_MODEL:
        return BACKUP_MODEL

    return PRIMARY_MODEL


def ask_ai(chat):
    """Ask Groq and automatically fallback if rate limited."""

    model_name = get_model_name()

    recent_messages = chat["messages"][-MAX_CONTEXT_MESSAGES:]

    langchain_messages = [
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
        for msg in recent_messages
    ]

    system_prompt = st.session_state.system_prompt.strip()

    messages = []

    if system_prompt:
        from langchain_core.messages import SystemMessage

        messages.append(
            SystemMessage(content=system_prompt)
        )

    messages.extend(langchain_messages)

    try:

        model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model_name,
            temperature=st.session_state.temperature,
        )

        response = model.invoke(messages)

        return response.content, model_name

    except RateLimitError:

        if model_name != BACKUP_MODEL:

            try:

                backup = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=BACKUP_MODEL,
                    temperature=st.session_state.temperature,
                )

                response = backup.invoke(messages)

                return response.content, BACKUP_MODEL

            except RateLimitError:
                raise

        raise


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:26px;
            font-weight:800;
            padding:8px 0 12px 0;
        ">
            ⚡ RacharlaGPT
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        st.session_state.auth_user.email
    )

    if st.button(
        "➕ New Chat",
        type="primary",
        use_container_width=True,
    ):

        create_chat()
        st.rerun()

    st.divider()

    search_text = st.text_input(
        "🔎 Search chats",
        placeholder="Search your history...",
    )

    st.markdown("### 💬 Your Chats")

    filtered_chats = list(
        st.session_state.chats.values()
    )

    if search_text.strip():

        query = search_text.lower().strip()

        filtered_chats = [
            chat
            for chat in filtered_chats
            if query in chat["title"].lower()
            or any(
                query in msg.get("content", "").lower()
                for msg in chat.get("messages", [])
            )
        ]

    filtered_chats.sort(
        key=lambda x: x.get("updated_at") or "",
        reverse=True,
    )

    for chat in filtered_chats:

        title = chat["title"] or "New Chat"

        if len(title) > 32:
            title = title[:32] + "..."

        is_current = (
            chat["id"]
            == st.session_state.current_chat_id
        )

        if st.button(
            ("🟢 " if is_current else "💬 ") + title,
            key="chat_" + chat["id"],
            use_container_width=True,
        ):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()

    st.divider()

    # =====================================================
    # RENAME
    # =====================================================

    current_chat = st.session_state.chats.get(
        st.session_state.current_chat_id
    )

    if current_chat:

        with st.expander("✏️ Rename current chat"):

            rename_value = st.text_input(
                "Chat name",
                value=current_chat["title"],
                key="rename_chat_title",
            )

            if st.button(
                "Save name",
                use_container_width=True,
            ):

                rename_value = " ".join(
                    rename_value.strip().split()
                )

                if rename_value:

                    current_chat["title"] = rename_value[:80]

                    save_chat(current_chat)

                    st.rerun()

    # =====================================================
    # DOWNLOAD
    # =====================================================

    if current_chat:

        export_lines = [
            f"# {current_chat['title']}\n"
        ]

        for msg in current_chat.get("messages", []):

            role = (
                "You"
                if msg.get("role") == "user"
                else "RacharlaGPT"
            )

            export_lines.append(
                f"## {role}\n\n"
                f"{msg.get('content', '')}\n"
            )

        st.download_button(
            "⬇️ Download Chat",
            data="\n".join(export_lines),
            file_name=(
                f"{current_chat['title'][:50] or 'chat'}.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )

    # =====================================================
    # AI SETTINGS
    # =====================================================

    with st.expander("⚙️ AI Settings"):

        model_choice = st.selectbox(
            "Model",
            [
                "Auto (recommended)",
                PRIMARY_MODEL,
                BACKUP_MODEL,
            ],
            index=[
                "Auto (recommended)",
                PRIMARY_MODEL,
                BACKUP_MODEL,
            ].index(st.session_state.selected_model),
        )

        st.session_state.selected_model = model_choice

        st.session_state.temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.2,
            value=float(st.session_state.temperature),
            step=0.1,
        )

        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=120,
        )

        if st.button(
            "Reset AI Settings",
            use_container_width=True,
        ):

            st.session_state.selected_model = (
                "Auto (recommended)"
            )

            st.session_state.temperature = 0.7

            st.session_state.system_prompt = (
                "You are RacharlaGPT, a helpful, intelligent "
                "and friendly AI assistant. Answer clearly "
                "and accurately. Keep responses natural and useful."
            )

            st.rerun()

    st.divider()

    # =====================================================
    # DELETE
    # =====================================================

    if current_chat:

        if st.button(
            "🗑️ Delete Current Chat",
            use_container_width=True,
        ):

            delete_chat(
                st.session_state.current_chat_id
            )

            st.rerun()

    if st.session_state.chats:

        if st.button(
            "🗑️ Delete All Chats",
            use_container_width=True,
        ):

            delete_all_chats()

            st.rerun()

    st.divider()

    if st.button(
        "🚪 Sign Out",
        use_container_width=True,
    ):

        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        st.session_state.auth_user = None
        st.session_state.chats = {}
        st.session_state.current_chat_id = None

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="rach-header">
        <div class="rach-logo">⚡</div>
        <div class="rach-title">RacharlaGPT</div>
        <div class="rach-subtitle">
            Intelligent conversations powered by Groq
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MAIN CHAT
# =========================================================

current_chat = st.session_state.chats.get(
    st.session_state.current_chat_id
)


if current_chat is None:

    st.info(
        "Start a new conversation by clicking "
        "**➕ New Chat**."
    )

    if st.button(
        "➕ Start New Chat",
        type="primary",
    ):

        create_chat()
        st.rerun()

else:

    # Chat title
    st.markdown(
        f"### {current_chat['title']}"
    )

    # Existing messages
    for message in current_chat["messages"]:

        role = message["role"]

        if role == "user":
            avatar = "👤"
        else:
            avatar = "⚡"

        with st.chat_message(
            role,
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

    # Input
    prompt = st.chat_input(
        "Message RacharlaGPT..."
    )

    if prompt:

        prompt = prompt.strip()

        if prompt:

            # User message
            current_chat["messages"].append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # First-message title
            if (
                current_chat["title"] == "New Chat"
                and len(
                    [
                        m
                        for m in current_chat["messages"]
                        if m["role"] == "user"
                    ]
                )
                == 1
            ):
                current_chat["title"] = generate_title(
                    prompt
                )

            save_chat(current_chat)

            # Display user immediately
            with st.chat_message(
                "user",
                avatar="👤",
            ):
                st.markdown(prompt)

            # AI response
            with st.chat_message(
                "assistant",
                avatar="⚡",
            ):

                with st.spinner(
                    "RacharlaGPT is thinking..."
                ):

                    try:

                        answer, used_model = ask_ai(
                            current_chat
                        )

                        st.markdown(answer)

                        current_chat["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        save_chat(current_chat)

                    except RateLimitError:

                        st.error(
                            "Groq rate limit reached. "
                            "Please wait a few minutes and try again."
                        )

                    except Exception as e:

                        st.error(
                            "Something went wrong while "
                            "generating the response."
                        )
