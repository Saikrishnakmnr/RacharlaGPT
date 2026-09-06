# 1. IMPORTS FIRST
import streamlit as st
from supabase import create_client, Client

# 2. INITIALIZE SUPABASE & SESSION STATES SECOND
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]

if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = st.session_state.supabase_client

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# 3. DEFINE RESTORE_SESSION FUNCTION THIRD
def restore_session():
    if st.session_state.get("auth_user"):
        return

    params = st.query_params
    code = params.get("code")

    if code:
        try:
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            if res and res.user:
                st.session_state.auth_user = res.user
                st.query_params.clear()
                st.rerun()
        except Exception:
            # Clear stale PKCE code parameter to prevent loop error
            st.query_params.pop("code", None)

# 4. CALL RESTORE_SESSION AFTER DEFINITION
restore_session()
