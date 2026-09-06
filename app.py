# ============================================================
# OAUTH CALLBACK & CODE EXCHANGE FIX
# ============================================================

def restore_session():
    """Handles session restoration and prevents PKCE code verifier errors."""
    if st.session_state.get("auth_user"):
        return

    params = st.query_params
    code = params.get("code")
    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")

    # 1. Direct restore via access/refresh tokens
    if access_token and refresh_token:
        try:
            res = supabase.auth.set_session(access_token, refresh_token)
            if res and res.user:
                st.session_state.auth_user = res.user
                return
        except Exception:
            pass

    # 2. Safe PKCE Code Exchange
    if code:
        try:
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            if res and res.user:
                st.session_state.auth_user = res.user
                st.query_params.clear()
                st.session_state.pop("google_oauth_error", None)
                st.session_state.pop("google_oauth_url", None)
                st.rerun()
                return
        except Exception as exc:
            # Clear stale code query parameter so the error screen vanishes
            st.session_state.google_oauth_error = str(exc)
            st.query_params.pop("code", None)
            st.query_params.pop("oauth_flow", None)

    # 3. Memory fallback
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.auth_user = session.user
    except Exception:
        pass


restore_session()
