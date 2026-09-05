import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from html import escape

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from supabase import create_client, Client


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "RacharlaGPT"

PRIMARY_MODEL = "openai/gpt-oss-20b"
BACKUP_MODEL = "llama-3.1-8b-instant"

MAX_CONTEXT_MESSAGES = 14

MODEL_OPTIONS = [
    "Auto (recommended)",
    PRIMARY_MODEL,
    BACKUP_MODEL,
]


DEFAULT_SYSTEM_PROMPT = """
You are RacharlaGPT, a helpful, intelligent, friendly and practical AI assistant.

IMPORTANT BEHAVIOR:

1. Answer the user's actual question directly.
2. If the user makes an obvious spelling mistake, typo, or phonetic mistake,
   understand the intended meaning and answer instead of stopping.
3. If a word is ambiguous and could have multiple meanings, briefly explain
   your interpretation and continue helping.
4. When the user asks "today", "tomorrow", "yesterday", "what day is it",
   "day of the week", "date today", or similar questions, use the current
   date and weekday supplied in the system context.
5. Never invent a special day, holiday, festival, observance, anniversary,
   or historical event. If the user asks about a "speciality" of today and
   there is no reliable information available in the supplied context,
   clearly say that there is no confirmed major observance rather than
   making something up.
6. If the user asks about today's date/day, give the exact date and weekday.
7. Use headings, bullet points, numbered steps and tables when useful.
8. If the user asks for code, provide complete working code.
9. Explain important code changes briefly.
10. Be concise for simple questions and detailed when the user needs detail.
11. If something is uncertain, say so clearly.
12. Do not stop answering merely because the user's spelling is imperfect.
13. Understand Indian English and common conversational spelling variations.
14. Be practical and solution-oriented.
"""


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "manatechsaavy_logo.png"
BANNER_PATH = ASSETS_DIR / "manatechsaavy_banner.jfif"


# ============================================================
# SECRETS
# ============================================================

def get_secret(name):
    value = os.environ.get(name)

    if value:
        return value.strip()

    try:
        value = st.secrets.get(name)

        if value:
            return str(value).strip()

    except Exception:
        pass

    return None


GROQ_API_KEY = get_secret("GROQ_API_KEY")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")


# ============================================================
# CONFIGURATION CHECK
# ============================================================

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase is not configured.")

    st.info(
        "Add SUPABASE_URL and SUPABASE_KEY in "
        "Streamlit Cloud → Manage app → Settings → Secrets."
    )

    st.stop()


if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================
# SUPABASE CLIENT
# ============================================================

if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )

supabase: Client = st.session_state.supabase_client


# ============================================================
# INDIA DATE / TIME CONTEXT
# ============================================================

def current_india_datetime():
    try:
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now(timezone.utc)


def date_context():
    now = current_india_datetime()

    return (
        f"Current date: {now.strftime('%d %B %Y')}\n"
        f"Current day of the week: {now.strftime('%A')}\n"
        f"Current time in India (IST): {now.strftime('%I:%M %p')}\n"
        f"Timezone: Asia/Kolkata (IST)"
    )


# ============================================================
# CSS — POLISHED UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.96);
    }

    section[data-testid="stSidebar"] {
        background: #f7f8fc;
        border-right: 1px solid #e7e9ef;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1rem;
    }


    /* ========================================================
       LOGIN / AUTH PAGE
       ======================================================== */

    .auth-page-title {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.15;
        color: #172033;
        margin: 0;
        letter-spacing: -0.8px;
    }

    .auth-page-subtitle {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.55;
        margin-top: 7px;
        margin-bottom: 18px;
    }

    .auth-small-note {
        color: #8a91a1;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 10px;
    }

    .auth-hero-title {
        font-size: 24px;
        font-weight: 800;
        color: #172033;
        margin-top: 12px;
        margin-bottom: 4px;
    }

    .auth-hero-text {
        color: #687083;
        font-size: 14px;
        line-height: 1.6;
        max-width: 560px;
        margin-bottom: 12px;
    }

    .auth-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eef4ff;
        color: #2457c5;
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Keep Streamlit image compact on auth page */
    .auth-banner-wrap {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    /* ========================================================
       MAIN BRANDING
       ======================================================== */

    .app-brand-name {
        font-size: 30px;
        font-weight: 800;
        color: #172033;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-top: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-top: 3px;
        margin-bottom: 10px;
    }

    /* ========================================================
       WELCOME
       ======================================================== */

    .welcome-box {
        width: 100%;
        max-width: 850px;
        margin: 9vh auto 3vh auto;
        text-align: center;
        padding: 20px;
    }

    .welcome-title {
        font-size: 31px;
        font-weight: 800;
        color: #172033;
    }

    .welcome-text {
        color: #6b7280;
        line-height: 1.6;
        max-width: 700px;
        margin: 10px auto;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-top: 0.8rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        .auth-page-title {
            font-size: 27px;
        }

        .auth-hero-title {
            font-size: 21px;
        }

        .auth-page-subtitle,
        .auth-hero-text {
            font-size: 13px;
        }

        .app-brand-name {
            font-size: 25px;
        }

        .app-subtitle {
            font-size: 13px;
        }

        .welcome-box {
            margin-top: 5vh;
        }

        .welcome-title {
            font-size: 25px;
        }

        .welcome-text {
            font-size: 14px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHENTICATION PAGE
# ============================================================

def show_auth():

    # --------------------------------------------------------
    # Desktop: compact two-column login experience.
    # Mobile: Streamlit automatically stacks the columns.
    # This prevents the banner from pushing the login far down.
    # --------------------------------------------------------

    left, right = st.columns(
        [1.25, 0.9],
        gap="large",
    )

    # ========================================================
    # LEFT — BRAND / BANNER
    # ========================================================

    with left:

        if LOGO_PATH.exists():
            logo_col1, logo_col2, logo_col3 = st.columns(
                [1, 1.5, 1]
            )

            with logo_col2:
                st.image(
                    str(LOGO_PATH),
                    width=82,
                )

        st.markdown(
            '<div class="auth-hero-title">RacharlaGPT</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="auth-hero-text">'
            'A practical AI assistant for learning, coding, '
            'ideas, research, and everyday questions.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<span class="auth-badge">⚡ Fast AI • ☁️ Saved Chats</span>',
            unsafe_allow_html=True,
        )

        st.write("")

        if BANNER_PATH.exists():
            st.markdown(
                '<div class="auth-banner-wrap">',
                unsafe_allow_html=True,
            )

            st.image(
                str(BANNER_PATH),
                width=560,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        else:
            st.info(
                "Banner image not found. "
                "Check assets/manatechsaavy_banner.jfif."
            )

    # ========================================================
    # RIGHT — LOGIN CARD
    # ========================================================

    with right:

        st.markdown(
            '<div class="auth-page-title">Welcome back 👋</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="auth-page-subtitle">'
            'Sign in to save and access your conversations permanently.'
            '</div>',
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(
            [
                "🔐 Sign In",
                "✨ Create Account",
            ]
        )

        # ====================================================
        # SIGN IN
        # ====================================================

        with login_tab:

            with st.form("login_form"):

                email = st.text_input(
                    "Email",
                    key="login_email",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Enter your password",
                )

                submit = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True,
                )

            if submit:

                if not email.strip() or not password:

                    st.warning(
                        "Enter your email and password."
                    )

                else:

                    try:

                        result = (
                            supabase.auth
                            .sign_in_with_password(
                                {
                                    "email": email.strip(),
                                    "password": password,
                                }
                            )
                        )

                        if result.user and result.session:

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Sign in failed. "
                                "Please check your email and password."
                            )

                    except Exception as exc:

                        st.error(
                            f"Sign in failed: {exc}"
                        )

            st.markdown(
                '<div class="auth-small-note">'
                'Your chat history is stored securely in Supabase.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ====================================================
        # CREATE ACCOUNT
        # ====================================================

        with signup_tab:

            with st.form("signup_form"):

                email = st.text_input(
                    "Email",
                    key="signup_email",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                    placeholder="At least 6 characters",
                )

                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_confirm",
                    placeholder="Repeat your password",
                )

                submit = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True,
                )

            if submit:

                if not email.strip() or not password:

                    st.warning(
                        "Enter your email and password."
                    )

                elif password != confirm:

                    st.warning(
                        "Passwords do not match."
                    )

                elif len(password) < 6:

                    st.warning(
                        "Password must be at least 6 characters."
                    )

                else:

                    try:

                        result = supabase.auth.sign_up(
                            {
                                "email": email.strip(),
                                "password": password,
                            }
                        )

                        if result.session and result.user:

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Account created. "
                                "You can now sign in."
                            )

                    except Exception as exc:

                        st.error(
                            f"Account creation failed: {exc}"
                        )


# ============================================================
# AUTH GATE
# ============================================================

if "auth_user" not in st.session_state:

    show_auth()
    st.stop()


auth_user = st.session_state.auth_user
USER_ID = str(auth_user.id)


# ============================================================
# MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_models(api_key):

    primary = ChatGroq(
        model=PRIMARY_MODEL,
        temperature=0.7,
        api_key=api_key,
    )

    backup = ChatGroq(
        model=BACKUP_MODEL,
        temperature=0.7,
        api_key=api_key,
    )

    return primary, backup


primary_llm, backup_llm = get_models(
    GROQ_API_KEY
)


# ============================================================
# SESSION STATE
# ============================================================

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "loaded_from_supabase" not in st.session_state:
    st.session_state.loaded_from_supabase = False

if "search" not in st.session_state:
    st.session_state.search = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Auto (recommended)"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7


# ============================================================
# CHAT FUNCTIONS
# ============================================================

def blank_chat():

    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "title": "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def load_chats():

    result = (
        supabase
        .table("chats")
        .select("*")
        .eq("user_id", USER_ID)
        .order("updated_at", desc=True)
        .execute()
    )

    chats = {}

    for row in result.data or []:

        chats[str(row["id"])] = {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "title": row.get("title") or "New Chat",
            "messages": row.get("messages") or [],
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    if not chats:

        chat = blank_chat()

        (
            supabase
            .table("chats")
            .insert(chat)
            .execute()
        )

        chats[chat["id"]] = chat

    return chats


def save_chat(chat):

    now = datetime.now(timezone.utc).isoformat()

    (
        supabase
        .table("chats")
        .update(
            {
                "title": chat["title"],
                "messages": chat["messages"],
                "updated_at": now,
            }
        )
        .eq("id", chat["id"])
        .eq("user_id", USER_ID)
        .execute()
    )

    chat["updated_at"] = now


def create_chat():

    chat = blank_chat()

    (
        supabase
        .table("chats")
        .insert(chat)
        .execute()
    )

    st.session_state.chats[chat["id"]] = chat
    st.session_state.current_chat_id = chat["id"]


def delete_chat(chat_id):

    (
        supabase
        .table("chats")
        .delete()
        .eq("id", chat_id)
        .eq("user_id", USER_ID)
        .execute()
    )

    st.session_state.chats.pop(
        chat_id,
        None,
    )

    if not st.session_state.chats:

        create_chat()

    else:

        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda x: x.get("updated_at") or "",
            reverse=True,
        )

        st.session_state.current_chat_id = ordered[0]["id"]


def delete_all():

    (
        supabase
        .table("chats")
        .delete()
        .eq("user_id", USER_ID)
        .execute()
    )

    st.session_state.chats = {}
    create_chat()


def title_for(text):

    text = " ".join(text.strip().split())

    if len(text) <= 34:
        return text

    return text[:34].rstrip() + "…"


def current_chat():

    if (
        st.session_state.current_chat_id
        not in st.session_state.chats
    ):

        create_chat()

    return st.session_state.chats[
        st.session_state.current_chat_id
    ]


# ============================================================
# AI RESPONSE
# ============================================================

def ask(chat):

    recent = chat["messages"][-MAX_CONTEXT_MESSAGES:]

    system_content = (
        st.session_state.system_prompt
        + "\n\n"
        + "REAL-TIME DATE CONTEXT:\n"
        + date_context()
    )

    msgs = [
        SystemMessage(content=system_content)
    ]

    for message in recent:

        if message["role"] == "user":

            msgs.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        else:

            msgs.append(
                AIMessage(
                    content=message["content"]
                )
            )

    selected = st.session_state.selected_model

    # --------------------------------------------------------
    # PRIMARY ONLY
    # --------------------------------------------------------

    if selected == PRIMARY_MODEL:

        try:

            llm = ChatGroq(
                model=PRIMARY_MODEL,
                temperature=st.session_state.temperature,
                api_key=GROQ_API_KEY,
            )

            return llm.invoke(msgs), "primary"

        except RateLimitError:

            llm = ChatGroq(
                model=BACKUP_MODEL,
                temperature=st.session_state.temperature,
                api_key=GROQ_API_KEY,
            )

            return llm.invoke(msgs), "backup"

    # --------------------------------------------------------
    # BACKUP ONLY
    # --------------------------------------------------------

    if selected == BACKUP_MODEL:

        llm = ChatGroq(
            model=BACKUP_MODEL,
            temperature=st.session_state.temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(msgs), "backup"

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    try:

        llm = ChatGroq(
            model=PRIMARY_MODEL,
            temperature=st.session_state.temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(msgs), "primary"

    except RateLimitError:

        llm = ChatGroq(
            model=BACKUP_MODEL,
            temperature=st.session_state.temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(msgs), "backup"


# ============================================================
# LOAD SUPABASE CHATS
# ============================================================

if not st.session_state.loaded_from_supabase:

    try:

        st.session_state.chats = load_chats()

        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda x: x.get("updated_at") or "",
            reverse=True,
        )

        if ordered:
            st.session_state.current_chat_id = ordered[0]["id"]

        st.session_state.loaded_from_supabase = True

    except Exception as exc:

        st.error(
            "Could not load chats from Supabase."
        )

        st.code(str(exc))
        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # LOGO + BRAND
    # --------------------------------------------------------

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            width=55,
        )

    st.markdown(
        """
        <div style="
            font-size:26px;
            font-weight:800;
            margin-bottom:15px;
        ">
            ⚡ RacharlaGPT
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        type="primary",
        use_container_width=True,
    ):

        create_chat()
        st.rerun()

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    user_email = getattr(
        auth_user,
        "email",
        "User",
    )

    st.caption(
        f"Signed in: {user_email}"
    )

    # --------------------------------------------------------
    # SIGN OUT
    # --------------------------------------------------------

    if st.button(
        "🚪 Sign Out",
        use_container_width=True,
    ):

        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        for key in [
            "auth_user",
            "chats",
            "current_chat_id",
            "loaded_from_supabase",
            "selected_model",
            "temperature",
        ]:

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    st.markdown("### YOUR CHATS")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "Search",
        placeholder="🔎 Search your chats...",
        label_visibility="collapsed",
    )

    st.session_state.search = search

    # --------------------------------------------------------
    # RENAME
    # --------------------------------------------------------

    if (
        st.session_state.current_chat_id
        in st.session_state.chats
    ):

        with st.expander("✏️ Rename current chat"):

            rename_value = st.text_input(
                "Chat name",
                value=st.session_state.chats[
                    st.session_state.current_chat_id
                ]["title"],
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

                    current = st.session_state.chats[
                        st.session_state.current_chat_id
                    ]

                    current["title"] = rename_value[:80]

                    save_chat(current)
                    st.rerun()

    # --------------------------------------------------------
    # CHAT LIST
    # --------------------------------------------------------

    ordered = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1].get("updated_at") or "",
        reverse=True,
    )

    for chat_id, item in ordered:

        q = search.lower().strip()

        matches = (
            not q
            or q in item["title"].lower()
            or any(
                q in str(
                    message.get("content", "")
                ).lower()
                for message in item["messages"]
            )
        )

        if not matches:
            continue

        c1, c2 = st.columns([4, 1])

        with c1:

            if st.button(
                f"💬 {item['title']}",
                key=f"open_{chat_id}",
                use_container_width=True,
            ):

                st.session_state.current_chat_id = chat_id
                st.rerun()

        with c2:

            if st.button(
                "🗑️",
                key=f"del_{chat_id}",
                use_container_width=True,
            ):

                delete_chat(chat_id)
                st.rerun()

    st.divider()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    current_export = st.session_state.chats.get(
        st.session_state.current_chat_id
    )

    if current_export:

        title = (
            current_export["title"]
            or "RacharlaGPT Chat"
        )

        txt_lines = [
            f"{title}\n",
            "RacharlaGPT Conversation\n",
            "=" * 50 + "\n",
        ]

        for message in current_export.get("messages", []):

            role = (
                "You"
                if message.get("role") == "user"
                else "RacharlaGPT"
            )

            txt_lines.append(f"\n{role}\n")
            txt_lines.append("-" * 30 + "\n")
            txt_lines.append(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )
            txt_lines.append("\n")

        text_download = "".join(txt_lines)

        # ----------------------------------------------------
        # SELF-CONTAINED HTML EXPORT
        # ----------------------------------------------------

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"<title>{escape(title)}</title>",
            """
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 900px;
                    margin: auto;
                    padding: 20px;
                    line-height: 1.6;
                    background: #ffffff;
                    color: #222222;
                }

                h1 {
                    font-size: 26px;
                }

                .message {
                    border: 1px solid #dddddd;
                    border-radius: 12px;
                    padding: 15px;
                    margin: 15px 0;
                }

                .role {
                    font-weight: bold;
                    margin-bottom: 8px;
                }

                .user {
                    background: #f5f5f5;
                }

                .assistant {
                    background: #ffffff;
                }

                .content {
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }

                .footer {
                    margin-top: 30px;
                    color: #777777;
                    font-size: 12px;
                    text-align: center;
                }
            </style>
            """,
            "</head>",
            "<body>",
            f"<h1>{escape(title)}</h1>",
        ]

        for message in current_export.get("messages", []):

            is_user = message.get("role") == "user"

            role = (
                "You"
                if is_user
                else "RacharlaGPT"
            )

            css_class = (
                "user"
                if is_user
                else "assistant"
            )

            content = escape(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )

            html_parts.append(
                f"""
                <div class="message {css_class}">
                    <div class="role">{role}</div>
                    <div class="content">{content}</div>
                </div>
                """
            )

        html_parts.extend(
            [
                """
                <div class="footer">
                    RacharlaGPT • Powered by Groq
                </div>
                """,
                "</body>",
                "</html>",
            ]
        )

        html_download = "".join(html_parts)

        st.markdown("### 📥 Download")

        st.download_button(
            "📄 Download TXT",
            data=text_download,
            file_name=f"{title[:45]}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.download_button(
            "🌐 Download HTML",
            data=html_download,
            file_name=f"{title[:45]}.html",
            mime="text/html",
            use_container_width=True,
        )

        st.caption(
            "TXT works everywhere. HTML keeps conversation formatting."
        )

    # --------------------------------------------------------
    # DELETE ALL
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete All Chats",
        use_container_width=True,
    ):

        delete_all()
        st.rerun()

    st.divider()

    # ========================================================
    # AI SETTINGS
    # ========================================================

    with st.expander(
        "⚙️ AI Settings",
        expanded=False,
    ):

        st.session_state.selected_model = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(
                st.session_state.selected_model
            ),
        )

        st.session_state.temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.2,
            value=float(
                st.session_state.temperature
            ),
            step=0.1,
            help=(
                "Lower = more focused. "
                "Higher = more creative."
            ),
        )

        st.session_state.system_prompt = st.text_area(
            "AI System Prompt",
            value=st.session_state.system_prompt,
            height=170,
        )

        if st.button(
            "↩️ Reset AI Settings",
            use_container_width=True,
        ):

            st.session_state.selected_model = (
                "Auto (recommended)"
            )

            st.session_state.temperature = 0.7

            st.session_state.system_prompt = (
                DEFAULT_SYSTEM_PROMPT
            )

            st.rerun()

    st.caption(f"Primary: {PRIMARY_MODEL}")
    st.caption(f"Fallback: {BACKUP_MODEL}")
    st.caption("☁️ Chats stored in Supabase")


# ============================================================
# MAIN APPLICATION BRANDING
# ============================================================

c1, c2 = st.columns([1, 12])

with c1:

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            width=48,
        )

    else:

        st.markdown("⚡")

with c2:

    st.markdown(
        """
        <div class="app-brand-name">
            RacharlaGPT
        </div>

        <div class="app-subtitle">
            Fast, intelligent AI conversations powered by Groq
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CURRENT CHAT
# ============================================================

chat = current_chat()


# ============================================================
# EMPTY CHAT
# ============================================================

if not chat["messages"]:

    st.markdown(
        """
        <div class="welcome-box">

            <div class="welcome-title">
                Welcome to RacharlaGPT
            </div>

            <div class="welcome-text">
                Your conversations are saved to your account.
                Ask questions, write code, brainstorm ideas,
                learn something new, or simply have a conversation.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in chat["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Message RacharlaGPT..."
)


if user_input:

    user_input = user_input.strip()

    if user_input:

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        chat["messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # ----------------------------------------------------
        # Automatic title
        # ----------------------------------------------------

        if chat["title"] == "New Chat":

            chat["title"] = title_for(
                user_input
            )

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                user_input
            )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "RacharlaGPT is thinking…"
                ):

                    response, used = ask(chat)

                answer = response.content

                if not isinstance(answer, str):

                    answer = str(answer)

                st.markdown(answer)

                if used == "backup":

                    st.caption(
                        "Primary model was rate-limited; "
                        f"answered with {BACKUP_MODEL}."
                    )

            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            save_chat(chat)

        except RateLimitError:

            with st.chat_message("assistant"):

                st.warning(
                    "Groq rate limit reached. "
                    "Please wait for the quota to reset "
                    "and try again."
                )

            chat["messages"].pop()

        except Exception as exc:

            with st.chat_message("assistant"):

                st.error(
                    "Something went wrong while "
                    "contacting Groq. Please try again."
                )

            chat["messages"].pop()

            print(
                "RacharlaGPT error:",
                type(exc).__name__,
                str(exc),
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9aa0ad;
        font-size:11px;
        margin-top:15px;
        padding-bottom:10px;
    ">
        RacharlaGPT • Powered by Groq • Chats stored in Supabase
    </div>
    """,
    unsafe_allow_html=True,
)
