import os
import uuid
import base64
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from supabase import create_client, Client


# ============================================================
# RACHARLAGPT
# Stable version:
# - Supabase authentication
# - Permanent chat history
# - Responsive logo/banner
# - Chat search
# - Rename
# - Delete
# - Download .txt
# - Current date/day awareness
# - Groq primary/fallback
# ============================================================


st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APP CONFIGURATION
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
You are RacharlaGPT, a helpful, intelligent, friendly and concise AI assistant.

IMPORTANT BEHAVIOR:

1. Always understand the user's intended meaning, even when there are:
   - spelling mistakes
   - typing mistakes
   - grammar mistakes
   - missing words
   - informal language

2. Do not stop answering just because the user has spelling mistakes.
   Infer the intended word from context and answer normally.

3. If a spelling correction is important, briefly mention the correction,
   but do not make the user repeat the question.

4. For questions involving today, tomorrow, yesterday, this week,
   day of the week, dates or current time:
   use the current date information supplied by the application.

5. Give accurate, clear and useful answers.

6. Use headings, bullet points, numbered steps and tables when they improve
   readability.

7. If the user asks for code, provide complete working code and briefly
   explain important changes.

8. Do not invent facts. If uncertain, clearly say so.

9. Be practical and solution-oriented.

10. Answer the user's actual question directly.
"""


# ============================================================
# CURRENT DATE / TIME
# ============================================================

def get_current_time():
    """
    Uses India Standard Time for reliable date/day answers.
    """

    try:
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now(timezone.utc)


CURRENT_TIME = get_current_time()

CURRENT_DATE_TEXT = CURRENT_TIME.strftime("%A, %d %B %Y")
CURRENT_TIME_TEXT = CURRENT_TIME.strftime("%I:%M %p")

DATE_CONTEXT = f"""
CURRENT DATE AND TIME INFORMATION

Today is: {CURRENT_DATE_TEXT}
Current time: {CURRENT_TIME_TEXT} IST

When the user asks:
- What day is today?
- Today's date?
- Today special?
- What is today?
- What day of the week?
- Tomorrow?
- Yesterday?

Use the current date above.

The user may make spelling or grammar mistakes.
Always infer the intended meaning and continue answering.
"""


# ============================================================
# PATHS / ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "manatechsaavy_logo.png"
BANNER_PATH = ASSETS_DIR / "manatechsaavy_banner.jfif"


# ============================================================
# SECRET HELPER
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
        "Open Streamlit Cloud → Manage app → Settings → Secrets "
        "and make sure SUPABASE_URL and SUPABASE_KEY exist."
    )

    st.stop()


if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")

    st.info(
        "Open Streamlit Cloud → Manage app → Settings → Secrets "
        "and make sure GROQ_API_KEY exists."
    )

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
# IMAGE HELPERS
# ============================================================

def image_as_base64(path):
    """
    Converts an image to a browser-friendly base64 string.
    This lets us control the exact responsive size with CSS.
    """

    try:
        if not path.exists():
            return None

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")

        suffix = path.suffix.lower()

        if suffix in [".jpg", ".jpeg", ".jfif"]:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"

        return f"data:{mime};base64,{encoded}"

    except Exception:
        return None


LOGO_DATA = image_as_base64(LOGO_PATH)
BANNER_DATA = image_as_base64(BANNER_PATH)


# ============================================================
# GLOBAL UI / RESPONSIVE CSS
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

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 5rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #f7f8fc;
        border-right: 1px solid #e7e9ef;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }

    .sidebar-logo {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        object-fit: cover;
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }

    .sidebar-brand-text {
        font-size: 22px;
        font-weight: 800;
        color: #172033;
    }

    .history-label {
        margin-top: 20px;
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

    div[data-testid="stSidebar"] button {
        border-radius: 10px !important;
    }


    /* ========================================================
       MAIN BRAND
       ======================================================== */

    .main-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2px 0 3px 0;
    }

    .main-logo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        object-fit: cover;
        box-shadow: 0 7px 20px rgba(0,0,0,0.12);
    }

    .main-brand-name {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #172033;
    }

    .main-subtitle {
        color: #6b7280;
        margin: 0 0 14px 60px;
        font-size: 14px;
    }


    /* ========================================================
       BANNER
       ======================================================== */

    .banner-wrapper {
        width: 100%;
        max-width: 920px;
        margin: 12px auto 18px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .banner-image {
        display: block;
        width: 100%;
        max-width: 920px;
        max-height: 220px;
        object-fit: contain;
        object-position: center;
        border-radius: 16px;
        box-shadow: 0 5px 22px rgba(0,0,0,0.08);
    }


    /* ========================================================
       AUTH PAGE
       ======================================================== */

    .auth-page {
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
    }

    .auth-banner-wrapper {
        width: 100%;
        max-width: 850px;
        margin: 4px auto 16px auto;
        display: flex;
        justify-content: center;
    }

    .auth-banner {
        width: 100%;
        max-width: 850px;
        max-height: 175px;
        object-fit: contain;
        object-position: center;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }

    .auth-header {
        text-align: center;
        margin: 4px auto 12px auto;
    }

    .auth-logo {
        width: 58px;
        height: 58px;
        border-radius: 17px;
        object-fit: cover;
        box-shadow: 0 7px 20px rgba(0,0,0,0.12);
        margin-bottom: 5px;
    }

    .auth-title {
        font-size: 30px;
        font-weight: 800;
        color: #172033;
        margin: 0;
    }

    .auth-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
    }


    /* ========================================================
       WELCOME
       ======================================================== */

    .welcome-card {
        max-width: 780px;
        margin: 7vh auto 2vh auto;
        text-align: center;
        padding: 20px 20px;
    }

    .welcome-logo-image {
        width: 70px;
        height: 70px;
        border-radius: 20px;
        object-fit: cover;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        margin-bottom: 10px;
    }

    .welcome-icon {
        font-size: 52px;
        margin-bottom: 5px;
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


    /* ========================================================
       SUGGESTIONS
       ======================================================== */

    .suggestion {
        border: 1px solid #e7e9ef;
        border-radius: 14px;
        padding: 14px 16px;
        background: #fbfcff;
        margin: 5px 0;
        text-align: left;
        color: #3f4654;
        min-height: 70px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-card {
        border: 1px solid #e7e9ef;
        border-radius: 13px;
        padding: 10px 12px;
        background: white;
        font-size: 12px;
        color: #596170;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .chat-footer-note {
        text-align: center;
        color: #9aa0ad;
        font-size: 11px;
        margin-top: 12px;
        padding-bottom: 8px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }

        .main-brand {
            gap: 9px;
        }

        .main-logo {
            width: 40px;
            height: 40px;
            border-radius: 11px;
        }

        .main-brand-name {
            font-size: 24px;
        }

        .main-subtitle {
            margin-left: 49px;
            font-size: 12px;
            margin-bottom: 9px;
        }

        .banner-wrapper {
            margin: 8px auto 12px auto;
        }

        .banner-image {
            width: 100%;
            max-height: 125px;
            border-radius: 10px;
        }

        .auth-page {
            max-width: 100%;
        }

        .auth-banner-wrapper {
            margin-bottom: 10px;
        }

        .auth-banner {
            max-height: 125px;
            border-radius: 10px;
        }

        .auth-logo {
            width: 50px;
            height: 50px;
            border-radius: 14px;
        }

        .auth-title {
            font-size: 25px;
        }

        .auth-subtitle {
            font-size: 13px;
        }

        .welcome-card {
            margin-top: 5vh;
            padding: 15px 10px;
        }

        .welcome-logo-image {
            width: 58px;
            height: 58px;
            border-radius: 17px;
        }

        .welcome-title {
            font-size: 26px;
        }

        .welcome-text {
            font-size: 14px;
        }

        .suggestion {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE HTML HELPERS
# ============================================================

def show_banner(auth=False):

    if not BANNER_DATA:
        return

    css_class = "auth-banner" if auth else "banner-image"
    wrapper_class = (
        "auth-banner-wrapper"
        if auth
        else "banner-wrapper"
    )

    st.markdown(
        f"""
        <div class="{wrapper_class}">
            <img
                class="{css_class}"
                src="{BANNER_DATA}"
                alt="ManaTechSaavy banner"
            >
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_logo(css_class="main-logo"):

    if not LOGO_DATA:
        return None

    return f"""
        <img
            class="{css_class}"
            src="{LOGO_DATA}"
            alt="RacharlaGPT logo"
        >
    """


# ============================================================
# AUTHENTICATION PAGE
# ============================================================

def show_auth():

    st.markdown('<div class="auth-page">', unsafe_allow_html=True)

    # Small responsive banner.
    # It will NEVER occupy the entire screen.
    show_banner(auth=True)

    logo_html = ""

    if LOGO_DATA:
        logo_html = show_logo("auth-logo")

    st.markdown(
        f"""
        <div class="auth-header">
            {logo_html}
            <div class="auth-title">Welcome to RacharlaGPT</div>
            <div class="auth-subtitle">
                Sign in to save your conversations permanently.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.8, 1])

    with center:

        login_tab, signup_tab = st.tabs(
            ["🔐 Sign In", "✨ Create Account"]
        )

        # ----------------------------------------------------
        # SIGN IN
        # ----------------------------------------------------

        with login_tab:

            with st.form("login_form"):

                email = st.text_input(
                    "Email",
                    key="login_email",
                    placeholder="Enter your email",
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

                if not email or not password:

                    st.warning(
                        "Please enter your email and password."
                    )

                else:

                    try:

                        result = supabase.auth.sign_in_with_password(
                            {
                                "email": email.strip(),
                                "password": password,
                            }
                        )

                        if result.user and result.session:

                            st.session_state.auth_user = result.user

                            st.rerun()

                        else:

                            st.error(
                                "Sign in failed. Please check your details."
                            )

                    except Exception as exc:

                        error_text = str(exc)

                        if "Invalid login credentials" in error_text:

                            st.error(
                                "Incorrect email or password."
                            )

                        else:

                            st.error(
                                f"Sign in failed: {error_text}"
                            )

        # ----------------------------------------------------
        # CREATE ACCOUNT
        # ----------------------------------------------------

        with signup_tab:

            with st.form("signup_form"):

                email = st.text_input(
                    "Email",
                    key="signup_email",
                    placeholder="Enter your email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                    placeholder="Create a password",
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

                if not email or not password:

                    st.warning(
                        "Please enter your email and password."
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

                            st.session_state.auth_user = result.user

                            st.rerun()

                        else:

                            st.success(
                                "Account created successfully. "
                                "Please sign in."
                            )

                    except Exception as exc:

                        error_text = str(exc)

                        if "already registered" in error_text.lower():

                            st.error(
                                "This email is already registered. "
                                "Please use Sign In."
                            )

                        else:

                            st.error(
                                f"Account creation failed: {error_text}"
                            )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#9aa0ad;
            font-size:11px;
            margin-top:18px;
        ">
            RacharlaGPT • Secure account • Permanent cloud chat history
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AUTH GATE
# ============================================================

# THIS IS IMPORTANT.
#
# A fresh browser session has no auth_user.
# Therefore the SIGN IN / CREATE ACCOUNT page is shown first.
#
# The main chatbot is impossible to open directly without login.

if "auth_user" not in st.session_state:

    show_auth()

    st.stop()


auth_user = st.session_state.auth_user

USER_ID = str(auth_user.id)


# ============================================================
# MODEL STATE
# ============================================================

if "system_prompt" not in st.session_state:

    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT


if "selected_model" not in st.session_state:

    st.session_state.selected_model = "Auto (recommended)"


if "temperature" not in st.session_state:

    st.session_state.temperature = 0.7


if "chats" not in st.session_state:

    st.session_state.chats = {}


if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = None


if "loaded_from_supabase" not in st.session_state:

    st.session_state.loaded_from_supabase = False


if "search" not in st.session_state:

    st.session_state.search = ""


# ============================================================
# CHAT HELPERS
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


def title_for(text):

    cleaned = " ".join(
        text.strip().split()
    )

    if not cleaned:

        return "New Chat"

    if len(cleaned) <= 34:

        return cleaned

    return cleaned[:34].rstrip() + "…"


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

        chat_id = str(row["id"])

        chats[chat_id] = {
            "id": chat_id,
            "user_id": str(row["user_id"]),
            "title": row.get("title") or "New Chat",
            "messages": row.get("messages") or [],
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    if not chats:

        chat = blank_chat()

        supabase.table("chats").insert(chat).execute()

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

        return

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
# LOAD USER CHATS
# ============================================================

if not st.session_state.loaded_from_supabase:

    try:

        st.session_state.chats = load_chats()

        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda x: x.get("updated_at") or "",
            reverse=True,
        )

        st.session_state.current_chat_id = ordered[0]["id"]

        st.session_state.loaded_from_supabase = True

    except Exception as exc:

        st.error(
            "Could not load your chats from Supabase."
        )

        st.code(str(exc))

        st.stop()


# ============================================================
# MODEL
# ============================================================

def ask(chat):

    recent = chat["messages"][
        -MAX_CONTEXT_MESSAGES:
    ]

    messages = [
        SystemMessage(
            content=(
                st.session_state.system_prompt
                + "\n\n"
                + DATE_CONTEXT
            )
        )
    ]

    for message in recent:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    selected = st.session_state.selected_model

    temperature = st.session_state.temperature


    # --------------------------------------------------------
    # Explicit primary
    # --------------------------------------------------------

    if selected == PRIMARY_MODEL:

        try:

            llm = ChatGroq(
                model=PRIMARY_MODEL,
                temperature=temperature,
                api_key=GROQ_API_KEY,
            )

            return llm.invoke(messages), "primary"

        except RateLimitError:

            llm = ChatGroq(
                model=BACKUP_MODEL,
                temperature=temperature,
                api_key=GROQ_API_KEY,
            )

            return llm.invoke(messages), "backup"


    # --------------------------------------------------------
    # Explicit backup
    # --------------------------------------------------------

    if selected == BACKUP_MODEL:

        llm = ChatGroq(
            model=BACKUP_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(messages), "backup"


    # --------------------------------------------------------
    # Automatic model
    # --------------------------------------------------------

    try:

        llm = ChatGroq(
            model=PRIMARY_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(messages), "primary"

    except RateLimitError:

        llm = ChatGroq(
            model=BACKUP_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )

        return llm.invoke(messages), "backup"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    logo_html = ""

    if LOGO_DATA:

        logo_html = show_logo(
            "sidebar-logo"
        )

    else:

        logo_html = (
            '<span style="font-size:25px;">⚡</span>'
        )


    st.markdown(
        f"""
        <div class="sidebar-brand">
            {logo_html}
            <span class="sidebar-brand-text">
                RacharlaGPT
            </span>
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
        key="new_chat_sidebar",
    ):

        create_chat()

        st.rerun()


    st.caption(
        f"Signed in: {getattr(auth_user, 'email', 'User')}"
    )


    # --------------------------------------------------------
    # SIGN OUT
    # --------------------------------------------------------

    if st.button(
        "🚪 Sign Out",
        use_container_width=True,
        key="signout_sidebar",
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
        ]:

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()


    st.markdown(
        '<div class="history-label">YOUR CHATS</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "Search chats",
        placeholder="🔎 Search your chats...",
        label_visibility="collapsed",
        key="chat_search_box",
    )

    st.session_state.search = search


    # --------------------------------------------------------
    # RENAME
    # --------------------------------------------------------

    if (
        st.session_state.current_chat_id
        in st.session_state.chats
    ):

        with st.expander(
            "✏️ Rename current chat"
        ):

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
                key="save_chat_name",
            ):

                rename_value = " ".join(
                    rename_value.strip().split()
                )

                if rename_value:

                    current = st.session_state.chats[
                        st.session_state.current_chat_id
                    ]

                    current["title"] = (
                        rename_value[:80]
                    )

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

    query = search.lower().strip()

    visible_count = 0

    for chat_id, item in ordered:

        matches = (
            not query
            or query in item["title"].lower()
            or any(
                query in str(
                    message.get("content", "")
                ).lower()
                for message in item["messages"]
            )
        )

        if not matches:

            continue

        visible_count += 1

        col_chat, col_delete = st.columns(
            [4, 1],
            gap="small",
        )

        is_current = (
            chat_id
            == st.session_state.current_chat_id
        )


        with col_chat:

            if st.button(
                f"💬 {item['title']}",
                key=f"open_{chat_id}",
                use_container_width=True,
                type=(
                    "primary"
                    if is_current
                    else "secondary"
                ),
            ):

                st.session_state.current_chat_id = (
                    chat_id
                )

                st.rerun()


        with col_delete:

            if st.button(
                "🗑️",
                key=f"delete_{chat_id}",
                use_container_width=True,
            ):

                delete_chat(chat_id)

                st.rerun()


    if visible_count == 0:

        st.markdown(
            '<div class="empty-history">'
            'No chats match your search.'
            '</div>',
            unsafe_allow_html=True,
        )


    st.divider()


    # --------------------------------------------------------
    # DOWNLOAD CHAT
    # --------------------------------------------------------

    current_export = st.session_state.chats.get(
        st.session_state.current_chat_id
    )

    if current_export:

        export_lines = []

        export_lines.append(
            "RACHARLAGPT"
        )

        export_lines.append(
            f"Chat: {current_export['title']}"
        )

        export_lines.append(
            f"Date exported: "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}"
        )

        export_lines.append(
            ""
        )

        export_lines.append(
            "=" * 60
        )

        export_lines.append(
            ""
        )


        for message in current_export.get(
            "messages",
            [],
        ):

            if message.get("role") == "user":

                role = "YOU"

            else:

                role = "RACHARLAGPT"


            export_lines.append(
                role
            )

            export_lines.append(
                "-" * 30
            )

            export_lines.append(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )

            export_lines.append(
                ""
            )

            export_lines.append(
                "=" * 60
            )

            export_lines.append(
                ""
            )


        export_text = "\n".join(
            export_lines
        )


        # TXT is deliberately used because it opens
        # on Android, iPhone, Windows, Mac and Linux.
        safe_filename = (
            current_export["title"]
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace("*", "-")
            .replace("?", "-")
            .replace('"', "-")
            .replace("<", "-")
            .replace(">", "-")
            .replace("|", "-")
            .strip()
        )

        if not safe_filename:

            safe_filename = "RacharlaGPT_Chat"


        st.download_button(
            "⬇️ Download Chat",
            data=export_text,
            file_name=(
                safe_filename[:50]
                + ".txt"
            ),
            mime="text/plain",
            use_container_width=True,
            key="download_chat_txt",
        )


    # --------------------------------------------------------
    # DELETE ALL
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete All Chats",
        use_container_width=True,
        key="delete_all_sidebar",
    ):

        delete_all()

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # AI SETTINGS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ AI Settings",
        expanded=False,
    ):

        st.session_state.selected_model = (
            st.selectbox(
                "Model",
                MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(
                    st.session_state.selected_model
                ),
            )
        )


        st.session_state.temperature = (
            st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.2,
                value=float(
                    st.session_state.temperature
                ),
                step=0.1,
                help=(
                    "Lower = focused and factual. "
                    "Higher = more creative."
                ),
            )
        )


        st.session_state.system_prompt = (
            st.text_area(
                "AI System Prompt",
                value=st.session_state.system_prompt,
                height=160,
            )
        )


        if st.button(
            "↩️ Reset AI Settings",
            use_container_width=True,
            key="reset_ai_settings",
        ):

            st.session_state.selected_model = (
                "Auto (recommended)"
            )

            st.session_state.temperature = 0.7

            st.session_state.system_prompt = (
                DEFAULT_SYSTEM_PROMPT
            )

            st.rerun()


    st.caption(
        f"Primary: {PRIMARY_MODEL}"
    )

    st.caption(
        f"Fallback: {BACKUP_MODEL}"
    )

    st.caption(
        "☁️ Chats stored permanently in Supabase"
    )


# ============================================================
# MAIN HEADER
# ============================================================

main_logo_html = ""

if LOGO_DATA:

    main_logo_html = show_logo(
        "main-logo"
    )

else:

    main_logo_html = (
        '<div style="'
        'width:48px;height:48px;'
        'border-radius:14px;'
        'display:flex;align-items:center;'
        'justify-content:center;'
        'background:#ff9f1c;'
        'font-size:25px;">⚡</div>'
    )


st.markdown(
    f"""
    <div class="main-brand">
        {main_logo_html}
        <div class="main-brand-name">
            RacharlaGPT
        </div>
    </div>

    <div class="main-subtitle">
        Fast, intelligent AI conversations powered by Groq
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SHOW COMPACT BANNER AFTER LOGIN
# ============================================================

show_banner(auth=False)


# ============================================================
# CURRENT CHAT
# ============================================================

chat = current_chat()


# ============================================================
# WELCOME SCREEN
# ============================================================

if not chat["messages"]:

    welcome_logo_html = ""

    if LOGO_DATA:

        welcome_logo_html = show_logo(
            "welcome-logo-image"
        )

    else:

        welcome_logo_html = (
            '<div class="welcome-icon">⚡</div>'
        )


    st.markdown(
        f"""
        <div class="welcome-card">

            {welcome_logo_html}

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


    suggestion_cols = st.columns(3)


    suggestions = [
        (
            "💡 Learn",
            "Explain a difficult topic simply",
        ),
        (
            "💻 Code",
            "Build a Streamlit application",
        ),
        (
            "🚀 Create",
            "Give me a creative project idea",
        ),
    ]


    for col, (
        heading,
        prompt_text,
    ) in zip(
        suggestion_cols,
        suggestions,
    ):

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


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

for message in chat["messages"]:

    with st.chat_message(
        message["role"]
    ):

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
        # SAVE USER MESSAGE
        # ----------------------------------------------------

        chat["messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )


        if chat["title"] == "New Chat":

            chat["title"] = title_for(
                user_input
            )


        with st.chat_message("user"):

            st.markdown(
                user_input
            )


        # ----------------------------------------------------
        # ASK AI
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "RacharlaGPT is thinking…"
                ):

                    response, model_used = ask(
                        chat
                    )


                answer = response.content


                if not isinstance(
                    answer,
                    str,
                ):

                    answer = str(answer)


                # --------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # --------------------------------------------

                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


                save_chat(chat)


                # --------------------------------------------
                # DISPLAY RESPONSE
                # --------------------------------------------

                st.markdown(
                    answer
                )


                if model_used == "backup":

                    st.caption(
                        "Primary model was temporarily "
                        "rate-limited. Answered using "
                        f"{BACKUP_MODEL}."
                    )


            except RateLimitError:

                friendly = """
⚠️ **Groq rate limit reached.**

The available Groq quota for the selected models is temporarily exhausted.

This is an API quota issue, not a problem with your question.

Please wait for the quota window to reset and try again.
"""

                st.markdown(
                    friendly
                )


                if (
                    chat["messages"]
                    and chat["messages"][-1]["role"]
                    == "user"
                ):

                    chat["messages"].pop()


            except Exception as exc:

                st.error(
                    "Something went wrong while contacting Groq. "
                    "Please try again."
                )


                print(
                    f"RacharlaGPT error: "
                    f"{type(exc).__name__}: {exc}"
                )


                if (
                    chat["messages"]
                    and chat["messages"][-1]["role"]
                    == "user"
                ):

                    chat["messages"].pop()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="chat-footer-note">
        RacharlaGPT • Powered by Groq •
        Chats stored securely in Supabase •
        AI can make mistakes, so verify important information.
    </div>
    """,
    unsafe_allow_html=True,
)
