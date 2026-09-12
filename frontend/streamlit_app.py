from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
IMAGES_DIR = Path("images")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

st.set_page_config(page_title="Agentic Content Orchestrator", page_icon="✨", layout="centered")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

/* ---------- shell ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(640px 320px at 88% -6%, rgba(124,58,237,.13), transparent 62%),
        radial-gradient(720px 360px at -8% 12%, rgba(79,70,229,.09), transparent 60%),
        #f4f5fa;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
/* ---------- page rhythm: one centered column, generous air ---------- */
.block-container {
    padding-top: 2.4rem !important;
    padding-bottom: 4rem !important;
    max-width: 880px !important;
}
.block-container > div > [data-testid="stVerticalBlock"] { width: 100%; }
[data-testid="stVerticalBlock"] { row-gap: .9rem; }
[data-testid="stHorizontalBlock"] { column-gap: .75rem; align-items: stretch; }

/* Main-area headers + eyebrow live in MORE_CSS (single source of truth). */

/* ---------- sidebar: dark history rail ---------- */
[data-testid="stSidebar"] {
    background: #0e1220 !important;
    border-right: 1px solid rgba(255,255,255,.07);
}
[data-testid="stSidebar"] * { color: #c9d1e8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.09) !important; margin: .8rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .05rem; }

.brand { display: flex; align-items: center; gap: .6rem; padding: .3rem .1rem .65rem; }
.brand-badge {
    width: 36px; height: 36px; border-radius: 11px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #8b5cf6);
    color: #fff !important; font-size: 1.1rem; font-weight: 800;
}
.brand-name { font-size: 1rem; font-weight: 700; color: #f2f4fb !important; line-height: 1.15; }
.brand-sub { font-size: .7rem; color: #8b93ad !important; }

[data-testid="stSidebar"] button {
    border: none !important; box-shadow: none !important; background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    justify-content: flex-start; text-align: left; width: 100%;
    border-radius: 11px !important; padding: .45rem .7rem !important;
    font-size: .855rem !important; font-weight: 500;
    transition: background .15s ease, color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,.08) !important; color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important; justify-content: center; font-weight: 600;
    box-shadow: 0 6px 18px rgba(79,70,229,.38) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }

.user-chip {
    display: flex; align-items: center; gap: .5rem;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.09);
    border-radius: 12px; padding: .5rem .7rem; font-size: .84rem; color: #dfe4f3 !important;
}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e7eaf3 !important;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(23,28,63,.06);
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

MORE_CSS = """
<style>
/* ---------- typography: single source of truth ---------- */
.eyebrow {
    display: block; font-size: .72rem !important; font-weight: 700 !important;
    letter-spacing: .12em !important; text-transform: uppercase;
    color: #6d5cf0 !important; margin: 0 0 .45rem !important; text-align: left !important; line-height: 1.4 !important;
}
.main-title { font-size: 2.15rem !important; font-weight: 800 !important; color: #141a33 !important; letter-spacing: -.02em; margin: 0 0 .3rem !important; text-align: left !important; line-height: 1.15 !important; }
.main-sub { color: #626b85 !important; font-size: .98rem !important; line-height: 1.6 !important; margin: 0 0 1.3rem !important; text-align: left !important; }
.article-title { font-size: 1.45rem !important; font-weight: 800 !important; color: #141a33 !important; letter-spacing: -.01em; margin: 0 !important; text-align: left !important; line-height: 1.25 !important; }
.meta-row { margin: .45rem 0 0 !important; text-align: left !important; }
.meta-chip {
    display: inline-block; background: #eef0f9; color: #4d566f !important;
    border-radius: 999px; padding: .18rem .65rem; font-size: .74rem !important; font-weight: 600 !important; margin-right: .4rem;
}

/* ---------- auth card ---------- */
.auth-card { padding: 1.8rem 1.8rem 1.6rem !important; }
.auth-kicker {
    font-size: .78rem !important; font-weight: 700 !important;
    letter-spacing: .12em !important; text-transform: uppercase;
    color: #6d5cf0 !important; text-align: center !important;
    margin-bottom: .75rem !important; line-height: 1.3 !important;
}
.auth-title { font-size: 1.35rem !important; font-weight: 800 !important; color: #141a33 !important; margin: .1rem 0 .15rem !important; text-align: center !important; line-height: 1.3 !important; }
.auth-sub { color: #626b85 !important; font-size: .9rem !important; margin: 0 0 1rem !important; text-align: center !important; line-height: 1.5 !important; }
.auth-badge {
    width: 48px; height: 48px; border-radius: 14px; margin: .3rem auto .8rem;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #8b5cf6);
    color: #fff !important; font-size: 1.3rem; font-weight: 800;
}
.auth-form-row { min-height: 3.1rem; padding: 0.78rem 1rem; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.auth-badge) [data-testid="stTextInput"] input {
    padding: .78rem 1rem !important; font-size: .95rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.auth-badge) [data-testid="stBaseButton-primary"] {
    min-height: 2.7rem; font-size: .95rem !important; font-weight: 700 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.auth-badge) [data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: .95rem !important; padding: .42rem 1.3rem !important;
}

/* ---------- inputs ---------- */
[data-testid="stTextArea"] textarea {
    border-radius: 14px !important; border: 1.5px solid #e3e7f2 !important;
    background: #fbfcff !important; color: #1c2340 !important; font-size: .95rem !important;
    padding: .8rem .9rem !important; transition: border .15s ease, box-shadow .15s ease;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6d5cf0 !important; box-shadow: 0 0 0 3px rgba(109,92,240,.16) !important; outline: none;
}
[data-testid="stTextArea"] textarea::placeholder { color: #9aa3bd !important; }

[data-testid="stTextInput"] input {
    border-radius: 12px !important; border: 1.5px solid #e3e7f2 !important;
    background: #fbfcff !important; color: #1c2340 !important; font-size: .95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6d5cf0 !important; box-shadow: 0 0 0 3px rgba(109,92,240,.15) !important;
}
[data-testid="stTextInput"] label, [data-testid="stDateInput"] label {
    font-weight: 600 !important; color: #3d466b !important; font-size: .84rem;
}

[data-testid="stDateInput"] input {
    border-radius: 11px !important; border: 1.5px solid #e3e7f2 !important;
    background: #fbfcff !important; color: #1c2340 !important; font-size: .86rem !important;
}

/* model popover pill */
[data-testid="stPopover"] button {
    border-radius: 999px !important; border: 1.5px solid #e3e7f2 !important;
    background: #ffffff !important; color: #3d466b !important;
    font-size: .84rem !important; font-weight: 600; padding: .42rem .9rem !important;
    box-shadow: none !important; transition: border .15s ease;
}
[data-testid="stPopover"] button:hover { border-color: #c7cbf5 !important; }

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; box-shadow: 0 6px 18px rgba(79,70,229,.30) !important;
    transition: filter .15s ease;
}
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
[data-testid="stBaseButton-secondary"] {
    border-radius: 12px !important; background: #fff !important;
    border: 1.5px solid #e3e7f2 !important; color: #3d466b !important; font-weight: 500;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #c7cbf5 !important; color: #4f46e5 !important; }

/* ---------- tabs as segmented control ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 5px; background: #edeff7; padding: 4px; border-radius: 13px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important; color: #5b6480 !important; font-weight: 600 !important;
    font-size: .88rem !important; background: transparent; border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #ffffff !important; color: #141a33 !important;
    box-shadow: 0 2px 8px rgba(23,28,63,.10);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ---------- misc ---------- */
[data-testid="stMarkdownContainer"] a { color: #4f46e5; }
[data-testid="stExpander"] details { border-radius: 12px !important; border: 1px solid #e7eaf3 !important; }
[data-testid="stDataFrame"] { border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 12px !important; }
</style>
"""
st.markdown(MORE_CSS, unsafe_allow_html=True)

def api_request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)
    # Auto-refresh on expired access token: swap the bearer and retry once.
    if response.status_code == 401 and st.session_state.get("refresh_token") and path != "/api/v1/auth/refresh":
        refresh = api_request(
            "POST",
            "/api/v1/auth/refresh",
            json={"refresh_token": st.session_state["refresh_token"]},
        )
        if refresh.ok:
            st.session_state.token = refresh.json()["access_token"]
            st.session_state.refresh_token = refresh.json()["refresh_token"]
            headers = kwargs.get("headers") or {}
            headers["Authorization"] = f"Bearer {st.session_state.token}"
            kwargs["headers"] = headers
            response = requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)
    return response


def show_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text or "Request failed."
    st.error(f"{response.status_code}: {detail}")


def safe_slug(title: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9 _-]", "", title.lower())
    return re.sub(r"\s+", "_", value).strip("_") or "blog"


def extract_title(markdown: str, fallback: str = "blog") -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def extract_evidence(markdown: str) -> list[dict[str, str]]:
    evidence = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^)\s]+", markdown):
        url = match.group(0).rstrip(".,")
        if url not in seen:
            seen.add(url)
            evidence.append({"url": url, "title": "Citation from generated article"})
    return evidence


def normalize_plan(plan: object) -> dict | None:
    if isinstance(plan, dict):
        return plan
    return None


def image_url(source: str) -> str:
    source = source.strip()
    return f"{API_BASE_URL}{source}" if source.startswith("/assets/") else source


def render_markdown(markdown: str) -> None:
    last = 0
    matches = list(IMAGE_RE.finditer(markdown))
    for match in matches:
        if match.start() > last:
            st.markdown(markdown[last : match.start()])
        st.image(image_url(match.group("src")), caption=match.group("alt") or None, use_container_width=True)
        last = match.end()
    if last < len(markdown):
        st.markdown(markdown[last:])


def bundle_bytes(markdown: str, title: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{safe_slug(title)}.md", markdown.encode("utf-8"))
        if IMAGES_DIR.exists():
            for path in IMAGES_DIR.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=str(path))
    return buffer.getvalue()

def signup_panel() -> None:
    with st.form("signup-form"):
        username = st.text_input("Username", placeholder="writer")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        username = (username or "").strip()
        password = password or ""
        if len(password) < 8:
            st.error("Password must be at least 8 characters. You entered " + str(len(password)) + ".")
            return
        if not re.match(r"^[A-Za-z0-9_.-]+$", username):
            st.error("Username may only contain letters, numbers, dots, dashes and underscores.")
            return
        response = api_request(
            "POST", "/api/v1/auth/signup", json={"username": username, "password": password}
        )
        if not response.ok:
            show_error(response)
            return
        # Seamless: sign up then immediately log in with the same credentials.
        login = api_request(
            "POST",
            "/api/v1/auth/token",
            data={"username": username, "password": password},
        )
        if login.ok:
            st.session_state.token = login.json()["access_token"]
            st.session_state.refresh_token = login.json()["refresh_token"]
            st.session_state.username = username
            st.session_state.signup_done = True
            st.rerun()
        else:
            st.session_state.signup_done = True
            st.session_state.signup_success = True
            st.session_state.new_username = username


def login_panel() -> None:
    if st.session_state.pop("signup_success", False):
        st.success(
            "Account created! The automatic sign-in hit a snag — please log in with the "
            "same username and password."
        )
    login_username = st.text_input(
        "Username",
        value=st.session_state.pop("new_username", st.session_state.get("login_username", "")),
    )
    login_password = st.text_input(
        "Password",
        type="password",
        key="login_password",
        value=st.session_state.get("login_password", ""),
    )
    submitted = st.button("Log in", type="primary", use_container_width=True)
    if submitted:
        login_username = (login_username or "").strip()
        response = api_request(
            "POST",
            "/api/v1/auth/token",
            data={"username": login_username, "password": login_password},
        )
        if response.ok:
            st.session_state.token = response.json()["access_token"]
            st.session_state.refresh_token = response.json()["refresh_token"]
            st.session_state.username = login_username
            st.session_state.pop("login_username", None)
            st.session_state.pop("login_password", None)
            st.rerun()
        else:
            st.session_state.login_username = login_username
            show_error(response)


def auth_screen() -> None:
    """Modern centered auth card (Claude/Notion style)."""
    _, mid, _ = st.columns([1, 4, 1], gap="large")
    # auth card
    with mid.container(border=True, height=420):
        st.markdown(
            '<p class="auth-kicker">Agentic Content Orchestrator</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="auth-badge">✦</div>', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">Welcome to Agentic Writer</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-sub">Research, draft and polish long-form articles — in one place.</p>',
            unsafe_allow_html=True,
        )
        tab_login, tab_signup = st.tabs(["Log in", "Create account"])
        with tab_login:
            login_panel()
        with tab_signup:
            signup_panel()

def history_rail(health: dict | None) -> None:
    """ChatGPT-style sidebar: brand, new-blog button, clickable history, user chip."""
    st.markdown(
        '<div class="brand"><div class="brand-badge">✦</div>'
        '<div><div class="brand-name">Agentic Writer</div>'
        '<div class="brand-sub">research → article</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("New blog", icon=":material/add_circle:", type="primary", use_container_width=True, key="new_blog"):
        for key in ("last_content", "last_job_id", "last_plan", "last_evidence", "last_timestamps"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption("HISTORY")

    token = st.session_state.get("token")
    try:
        history = api_request("GET", "/api/v1/blogs", headers={"Authorization": f"Bearer {token}"})
        blogs = history.json().get("blogs", []) if history.ok else []
    except Exception:
        blogs = []

    if not blogs:
        st.caption("No blogs yet. Articles you generate will show up here.")
    else:
        for blog in blogs:
            created = (blog.get("created_at") or "")[:10]
            label = f"{blog['title']}  ·  {created}" if created else blog["title"]
            if st.button(
                label,
                icon=":material/description:",
                key=f"hist_{blog['job_id']}",
                use_container_width=True,
                help="Open this article",
            ):
                result = api_request(
                    "GET",
                    f"/api/v1/jobs/{blog['job_id']}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if result.ok:
                    job = result.json()
                    st.session_state.last_content = job.get("content", "")
                    st.session_state.last_job_id = blog["job_id"]
                    st.session_state.last_plan = job.get("plan")
                    st.session_state.last_evidence = job.get("evidence", [])
                    st.rerun()
                else:
                    show_error(result)

    st.divider()
    username = st.session_state.get("username") or "writer"
    st.markdown(f'<div class="user-chip">👤&nbsp; {username}</div>', unsafe_allow_html=True)
    if st.button("Log out", icon=":material/logout:", key="logout", use_container_width=True):
        for key in ("token", "refresh_token", "username", "last_content", "last_job_id",
                    "last_plan", "last_evidence", "last_timestamps"):
            st.session_state.pop(key, None)
        st.rerun()
    status = "🟢 backend online" if health else "🔴 backend offline"
    st.caption(f"{status} · {API_BASE_URL}")


def composer() -> None:
    """Claude/ChatGPT-style prompt box with an inline model picker."""
    if not st.session_state.get("last_content"):
        st.markdown(
            '<p class="main-title">What should we write about?</p>'
            '<p class="main-sub">Give a topic, pick a model, and the research-to-article pipeline handles the rest.</p>',
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        topic = st.text_area(
            "Topic",
            label_visibility="collapsed",
            placeholder="e.g. How should production RAG systems be evaluated?",
            height=128,
            key="composer_topic",
        )
        cols = st.columns([1.5, 1, 1], vertical_alignment="center")
        options = ["Auto (fallback chain)"] + list(st.session_state.get("_model_options") or [])
        current = st.session_state.get("model_choice") or options[0]
        if current not in options:
            current = options[0]
        with cols[0]:
            with st.popover(f"✦ Model · {current.replace('gemini/', '')}", use_container_width=True):
                st.caption(
                    "The model that drafts and reviews your article. If its quota runs out, "
                    "the fallback chain takes over automatically."
                )
                st.radio("Model", options, key="model_choice", label_visibility="collapsed")
        with cols[1]:
            as_of = st.date_input(
                "As-of date",
                value=None,
                help="Optional. Limits web research to this date or earlier.",
            )
        with cols[2]:
            submitted = st.button("Generate", icon=":material/send:", type="primary", use_container_width=True)
    if submitted:
        run_generation((topic or "").strip(), as_of, current)

def run_generation(topic: str, as_of, model_choice: str) -> None:
    """Submit the job and poll until the article is ready (same contract as before)."""
    token = st.session_state.get("token")
    if not topic:
        st.warning("Please enter a topic first.")
        return
    payload = {
        "topic": topic,
        "preferred_model": None if model_choice.startswith("Auto") else model_choice,
    }
    if as_of is not None:
        payload["as_of"] = as_of.isoformat()
    response = api_request("POST", "/api/v1/generate", headers={"Authorization": f"Bearer {token}"}, json=payload)
    if not response.ok:
        show_error(response)
        return
    job_id = response.json()["job_id"]
    progress = st.progress(0, text="Starting workflow...")
    status_box = st.status("Running the pipeline...", expanded=True)
    stages = ["router", "research", "planner", "workers", "quality gate", "images"]
    shown: set[str] = set()
    for attempt in range(60):
        time.sleep(2)
        result = api_request("GET", f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        if not result.ok:
            progress.empty()
            show_error(result)
            return
        job = result.json()
        if job["status"] == "completed":
            st.session_state.last_content = job.get("content", "")
            st.session_state.last_job_id = job_id
            st.session_state.last_plan = job.get("plan")
            st.session_state.last_evidence = job.get("evidence", [])
            st.session_state.last_timestamps = (job.get("created_at"), job.get("updated_at"))
            progress.progress(100, text="Article ready")
            status_box.update(label="Done", state="complete", expanded=False)
            st.rerun()
        if job["status"] == "failed":
            progress.empty()
            status_box.update(label="Generation failed", state="error", expanded=True)
            st.error(job.get("error") or "Article generation failed.")
            return
        # Real workflow stage reported by the backend (falls back to a heuristic).
        stage = job.get("stage") or stages[min(len(stages) - 1, attempt // 10)]
        if stage not in shown:
            status_box.write(f"Stage: `{stage}`")
            shown.add(stage)
        progress.progress(min(95, (attempt + 1) * 95 // 60), text=f"{job['status']} · {stage}")
    else:
        progress.empty()
        st.warning(f"Still running. Job ID: {job_id}")

def article_view() -> None:
    """Rendered article with plan/evidence/preview/images/logs tabs."""
    import html as html_mod

    markdown = st.session_state.get("last_content", "")
    if not markdown:
        return
    title = extract_title(markdown)
    st.markdown(f'<p class="article-title">{html_mod.escape(title)}</p>', unsafe_allow_html=True)
    created, updated = st.session_state.get("last_timestamps", (None, None))
    chips = (
        '<span class="meta-chip">✅ completed</span>'
        f'<span class="meta-chip">job {str(st.session_state.get("last_job_id"))[:8]}</span>'
    )
    if created:
        chips += f'<span class="meta-chip">created {str(created)[:10]}</span>'
    if updated:
        chips += f'<span class="meta-chip">updated {str(updated)[:10]}</span>'
    st.markdown(chips, unsafe_allow_html=True)
    st.write("")

    plan_tab, evidence_tab, preview_tab, images_tab, logs_tab = st.tabs(
        ["Plan", "Evidence", "Markdown Preview", "Images", "Logs"]
    )
    with plan_tab:
        plan = normalize_plan(st.session_state.get("last_plan"))
        if plan:
            st.write("**Title:**", plan.get("blog_title", title))
            columns = st.columns(3)
            columns[0].write("**Audience:** " + str(plan.get("audience", "")))
            columns[1].write("**Tone:** " + str(plan.get("tone", "")))
            columns[2].write("**Blog kind:** " + str(plan.get("blog_kind", "")))
            tasks = plan.get("tasks", [])
            if tasks:
                st.dataframe(
                    [
                        {
                            "id": task.get("id"),
                            "title": task.get("title"),
                            "target_words": task.get("target_words"),
                            "requires_research": task.get("requires_research", False),
                            "requires_citations": task.get("requires_citations", False),
                            "requires_code": task.get("requires_code", False),
                            "tags": ", ".join(task.get("tags") or []),
                        }
                        for task in tasks
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander("Task details"):
                    st.json(tasks)
        else:
            st.info("Plan metadata is available for blogs generated after the backend update.")
        st.caption(f"Job: {st.session_state.get('last_job_id')} | Status: completed")
    with evidence_tab:
        evidence = st.session_state.get("last_evidence") or extract_evidence(markdown)
        if evidence:
            st.dataframe(
                [
                    {
                        "title": item.get("title", "Source"),
                        "published_at": item.get("published_at"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                    }
                    for item in evidence
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No citations were included in this article. Closed-book topics may not require web evidence.")
    with preview_tab:
        render_markdown(markdown)
        st.download_button("Download Markdown", markdown.encode("utf-8"), f"{safe_slug(title)}.md", "text/markdown")
        st.download_button(
            "Download Bundle (MD + images)",
            bundle_bytes(markdown, title),
            f"{safe_slug(title)}_bundle.zip",
            "application/zip",
        )
    with images_tab:
        sources = [image_url(match.group("src")) for match in IMAGE_RE.finditer(markdown)]
        if not sources:
            st.info("No images were generated for this article.")
        for source in sources:
            st.image(source, use_container_width=True)
    with logs_tab:
        st.code(
            f"API: {API_BASE_URL}\nJob: {st.session_state.get('last_job_id')}\n"
            f"Status: completed\nEvidence found: {len(st.session_state.get('last_evidence') or extract_evidence(markdown))}\n"
            f"Images found: {len(IMAGE_RE.findall(markdown))}\n"
            f"Created: {st.session_state.get('last_timestamps', (None, None))[0]}\n"
            f"Updated: {st.session_state.get('last_timestamps', (None, None))[1]}"
        )

# -----------------------------
# App flow
# -----------------------------
if st.session_state.pop("signup_done", False):
    st.toast("Welcome! Your account is ready — you're signed in.", icon="✨")

try:
    health_response = api_request("GET", "/api/v1/health")
    health = health_response.json() if health_response.ok else None
except requests.RequestException:
    health = None

# Fetch available LLM candidates once per session for the model picker.
if "_model_options" not in st.session_state:
    try:
        models_response = api_request("GET", "/api/v1/models")
        st.session_state._model_options = models_response.json().get("models", []) if models_response.ok else []
    except requests.RequestException:
        st.session_state._model_options = []

if st.session_state.get("token"):
    with st.sidebar:
        history_rail(health)
    composer()
    article_view()
else:
    auth_screen()







