from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGES_DIR = Path("images")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

st.set_page_config(page_title="Agentic Writer", page_icon="🖋️", layout="centered")

# =========================================================================
# DESIGN — "Ink & Quill"
# A writing tool should feel like a writing tool, not another indigo SaaS
# dashboard. The app chrome is a dark ink-navy, chat-app layout (dark
# sidebar + dark canvas, like the interface you already live in daily).
# The one bold move: a generated article renders on an actual warm paper
# card inside that dark chrome — the manuscript sitting on the desk. A
# brass/gold accent (not violet, not Claude's terracotta) marks the one
# interactive thread running through both. Headlines use a serif
# (Lora) for an editorial voice; everything interactive stays in Inter.
# Icons are plain text/characters, not emoji or font-ligatures, so
# there is no font-load path that can turn an icon into stray text.
# =========================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Lora:ital,wght@0,500;0,600;0,700;1,500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

/* ---------- app shell: ink navy ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 480px at 85% -10%, rgba(201,151,76,.08), transparent 60%),
        radial-gradient(700px 420px at 0% 100%, rgba(120,140,200,.05), transparent 55%),
        #10121b;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2.3rem !important;
    padding-bottom: 4rem !important;
    max-width: 860px !important;
}
.block-container > div > [data-testid="stVerticalBlock"] { width: 100%; }
[data-testid="stVerticalBlock"] { row-gap: .85rem; }
[data-testid="stHorizontalBlock"] { column-gap: .7rem; align-items: stretch; }

h1,h2,h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family: 'Lora', Georgia, serif; }

/* ---------- sidebar: darkest layer, chat-history feel ---------- */
[data-testid="stSidebar"] {
    background: #0a0b12 !important;
    border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: #b7bdd4 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.07) !important; margin: .85rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .05rem; }

.brand { display: flex; align-items: center; gap: .65rem; padding: .15rem .1rem .8rem; }
.brand-mark {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.05rem;
    background: linear-gradient(160deg, #2a2d3d, #171923);
    border: 1px solid rgba(201,151,76,.35);
    color: #d9ab5c !important;
}
.brand-name { font-family: 'Lora', Georgia, serif; font-size: 1.05rem; font-weight: 600; color: #f1ede2 !important; line-height: 1.15; }
.brand-sub { font-size: .7rem; color: #6d7390 !important; }

.rail-label {
    font-size: .74rem !important; font-weight: 600 !important; color: #757ea3 !important;
    margin: .1rem 0 .5rem !important;
}

[data-testid="stSidebar"] button {
    border: none !important; box-shadow: none !important; background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    justify-content: flex-start !important; text-align: left; width: 100%;
    border-radius: 9px !important; padding: .5rem .65rem !important;
    font-size: .84rem !important; font-weight: 500; line-height: 1.35 !important;
    transition: background .15s ease, color .15s ease;
    white-space: pre-line; overflow-wrap: anywhere;
    border-left: 2px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,.05) !important; color: #f1ede2 !important;
}
.hist-active [data-testid="stBaseButton-secondary"] {
    background: rgba(201,151,76,.09) !important;
    border-left: 2px solid #c9974c !important;
    color: #f1ede2 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #c9974c !important;
    color: #171923 !important; justify-content: center !important; font-weight: 700 !important;
    border-radius: 9px !important; padding: .55rem .7rem !important; font-size: .86rem !important;
    box-shadow: 0 4px 14px rgba(201,151,76,.25) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover { background: #d9ab5c !important; }

.hist-empty { font-size: .8rem; color: #5c6280; line-height: 1.55; padding: .3rem .1rem .5rem; }

.user-chip {
    display: flex; align-items: center; gap: .55rem;
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; padding: .5rem .65rem; font-size: .83rem; color: #cfd3e6 !important;
}
.user-chip .avatar {
    width: 22px; height: 22px; border-radius: 50%; background: #c9974c; color: #171923 !important;
    display: flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 700; flex-shrink: 0;
}
.status-line { display: flex; align-items: center; gap: .45rem; font-size: .72rem !important; color: #565c78 !important; margin-top: .55rem; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.on  { background: #5cc98a; box-shadow: 0 0 0 3px rgba(92,201,138,.16); }
.status-dot.off { background: #e15b5b; box-shadow: 0 0 0 3px rgba(225,91,91,.16); }

/* ---------- main canvas cards: dark ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #171a26;
    border: 1px solid #262b3b !important;
    border-radius: 14px;
}

/* ---------- hero / composer copy ---------- */
.hero { margin: .2rem 0 1.15rem; }
.hero-title {
    font-family: 'Lora', Georgia, serif; font-style: italic;
    font-size: 1.95rem !important; font-weight: 600 !important; color: #f1ede2 !important;
    margin: 0 !important; line-height: 1.25 !important;
}
.hero-sub { color: #7d84a3 !important; font-size: .93rem !important; line-height: 1.55 !important; margin: .35rem 0 0 !important; }

/* ---------- composer (chat-input styled) ---------- */
[data-testid="stTextArea"] textarea {
    border-radius: 12px !important; border: 1.5px solid #262b3b !important;
    background: #12141e !important; color: #eceef5 !important; font-size: .96rem !important;
    padding: .85rem .95rem !important; transition: border .15s ease, box-shadow .15s ease;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #c9974c !important; box-shadow: 0 0 0 3px rgba(201,151,76,.16) !important; outline: none;
}
[data-testid="stTextArea"] textarea::placeholder { color: #565c78 !important; }

[data-testid="stTextInput"] input {
    border-radius: 10px !important; border: 1.5px solid #262b3b !important;
    background: #12141e !important; color: #eceef5 !important; font-size: .95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c9974c !important; box-shadow: 0 0 0 3px rgba(201,151,76,.15) !important;
}
[data-testid="stTextInput"] label, [data-testid="stDateInput"] label {
    font-weight: 600 !important; color: #9aa0bd !important; font-size: .8rem;
}
[data-testid="stDateInput"] input {
    border-radius: 9px !important; border: 1.5px solid #262b3b !important;
    background: #12141e !important; color: #eceef5 !important; font-size: .85rem !important;
}

[data-testid="stPopover"] button {
    border-radius: 999px !important; border: 1.5px solid #262b3b !important;
    background: #12141e !important; color: #cfd3e6 !important;
    font-size: .83rem !important; font-weight: 600; padding: .42rem .9rem !important;
    box-shadow: none !important; transition: border .15s ease;
}
[data-testid="stPopover"] button:hover { border-color: #4a4160 !important; }
[data-testid="stPopoverBody"] { background: #171a26 !important; border: 1px solid #262b3b !important; }
[data-testid="stPopoverBody"] * { color: #cfd3e6 !important; }

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {
    background: #c9974c !important;
    color: #171923 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; box-shadow: 0 6px 16px rgba(201,151,76,.22) !important;
    transition: background .15s ease;
}
[data-testid="stBaseButton-primary"]:hover { background: #d9ab5c !important; }
[data-testid="stBaseButton-secondary"] {
    border-radius: 10px !important; background: #12141e !important;
    border: 1.5px solid #262b3b !important; color: #cfd3e6 !important; font-weight: 500;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #4a4160 !important; color: #f1ede2 !important; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; background: #12141e; padding: 4px; border-radius: 11px; border: 1px solid #262b3b;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important; color: #7d84a3 !important; font-weight: 600 !important;
    font-size: .86rem !important; background: transparent; border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #262b3b !important; color: #f1ede2 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] p { color: #cfd3e6 !important; }

/* ---------- the manuscript: the one bold, warm element ---------- */
.paper {
    background: #f6f0e3;
    border: 1px solid #e3d7bb;
    border-radius: 4px 4px 10px 10px;
    border-top: 3px solid #c9974c;
    padding: 1.9rem 2rem 1.6rem;
    box-shadow: 0 18px 40px rgba(0,0,0,.28);
}
.paper .article-title {
    font-family: 'Lora', Georgia, serif; font-size: 1.6rem !important; font-weight: 700 !important;
    color: #2a2620 !important; letter-spacing: -.01em; margin: 0 !important; line-height: 1.3 !important;
}
.paper .byline {
    font-size: .78rem !important; color: #8a7f63 !important; margin: .4rem 0 0 !important;
    display: flex; gap: .55rem; flex-wrap: wrap; align-items: center;
}
.paper .byline .status-pill {
    background: #e4efe4; color: #35703f !important; border-radius: 999px;
    padding: .15rem .62rem; font-size: .72rem !important; font-weight: 700;
}
.paper .stat-line {
    display: flex; margin: 1rem 0 .2rem; border-top: 1px solid #e3d7bb; border-bottom: 1px solid #e3d7bb;
}
.paper .stat-block {
    flex: 1; padding: .55rem 0; text-align: center; border-right: 1px solid #e3d7bb;
}
.paper .stat-block:last-child { border-right: none; }
.paper .stat-num { font-family: 'Lora', Georgia, serif; font-size: 1.2rem; font-weight: 700; color: #2a2620; }
.paper .stat-lbl { font-size: .68rem; color: #8a7f63; font-weight: 600; }
.paper [data-testid="stMarkdownContainer"] p,
.paper [data-testid="stMarkdownContainer"] li { color: #2a2620 !important; font-size: .98rem; line-height: 1.7; }
.paper [data-testid="stMarkdownContainer"] h1,
.paper [data-testid="stMarkdownContainer"] h2,
.paper [data-testid="stMarkdownContainer"] h3 { color: #201c15 !important; }
.paper [data-testid="stMarkdownContainer"] a { color: #a5702a; }
.paper [data-testid="stDownloadButton"] button {
    background: #2a2620 !important; color: #f6f0e3 !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.paper [data-testid="stDownloadButton"] button:hover { background: #443d31 !important; }

/* ---------- auth ---------- */
.auth-kicker { font-size: .76rem !important; font-weight: 600 !important; color: #c9974c !important; text-align: center !important; margin-bottom: .6rem !important; }
.auth-mark {
    width: 42px; height: 42px; border-radius: 12px; margin: .2rem auto .7rem;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    background: linear-gradient(160deg, #2a2d3d, #171923); border: 1px solid rgba(201,151,76,.35);
}
.auth-title { font-family: 'Lora', Georgia, serif; font-style: italic; font-size: 1.4rem !important; font-weight: 600 !important; color: #f1ede2 !important; margin: 0 0 .15rem !important; text-align: center !important; }
.auth-sub { color: #7d84a3 !important; font-size: .89rem !important; margin: 0 0 1rem !important; text-align: center !important; line-height: 1.5 !important; }

/* ---------- misc ---------- */
[data-testid="stMarkdownContainer"] a { color: #c9974c; }
[data-testid="stExpander"] details { border-radius: 10px !important; border: 1px solid #262b3b !important; background: #12141e; }
[data-testid="stExpander"] summary p { color: #cfd3e6 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stCaptionContainer"] { color: #6d7390 !important; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# =========================================================================
# BACKEND HELPERS — unchanged contract with your FastAPI service
# =========================================================================
def api_request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)
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


# =========================================================================
# AUTH SCREENS
# =========================================================================
def signup_panel() -> None:
    with st.form("signup-form"):
        username = st.text_input("Username", placeholder="writer")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        username = (username or "").strip()
        password = password or ""
        if len(password) < 8:
            st.error(f"Password must be at least 8 characters. You entered {len(password)}.")
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
    """Centered auth card, matching the ink & quill theme."""
    _, mid, _ = st.columns([1, 4, 1], gap="large")
    with mid.container(border=True):
        st.markdown('<p class="auth-kicker">Agentic Writer</p>', unsafe_allow_html=True)
        st.markdown('<div class="auth-mark">🖋️</div>', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">Sit down, the desk is ready</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-sub">Research, draft and polish long-form articles — in one place.</p>',
            unsafe_allow_html=True,
        )
        tab_login, tab_signup = st.tabs(["Log in", "Create account"])
        with tab_login:
            login_panel()
        with tab_signup:
            signup_panel()


# =========================================================================
# SIDEBAR — chat-style history rail
# =========================================================================
def history_rail(health: dict | None) -> None:
    st.markdown(
        '<div class="brand"><div class="brand-mark">🖋️</div>'
        '<div><div class="brand-name">Agentic Writer</div>'
        '<div class="brand-sub">research → article</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("+  New blog", type="primary", use_container_width=True, key="new_blog"):
        for key in ("last_content", "last_job_id", "last_plan", "last_evidence", "last_timestamps"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.markdown('<p class="rail-label">History</p>', unsafe_allow_html=True)

    token = st.session_state.get("token")
    try:
        history = api_request("GET", "/api/v1/blogs", headers={"Authorization": f"Bearer {token}"})
        blogs = history.json().get("blogs", []) if history.ok else []
    except requests.RequestException:
        blogs = []

    if not blogs:
        st.markdown(
            '<p class="hist-empty">No blogs yet. Articles you generate will show up here.</p>',
            unsafe_allow_html=True,
        )
    else:
        active_job = st.session_state.get("last_job_id")
        for blog in blogs:
            created = (blog.get("created_at") or "")[:10]
            title = blog["title"] if len(blog["title"]) <= 40 else blog["title"][:37] + "..."
            label = f"{title}\n{created}" if created else title
            wrapper_class = "hist-active" if blog["job_id"] == active_job else ""
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            clicked = st.button(label, key=f"hist_{blog['job_id']}", use_container_width=True, help="Open this article")
            st.markdown("</div>", unsafe_allow_html=True)
            if clicked:
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
    initial = username[:1].upper()
    st.markdown(
        f'<div class="user-chip"><div class="avatar">{initial}</div>{username}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Log out", key="logout", use_container_width=True):
        for key in ("token", "refresh_token", "username", "last_content", "last_job_id",
                    "last_plan", "last_evidence", "last_timestamps"):
            st.session_state.pop(key, None)
        st.rerun()

    dot_class = "on" if health else "off"
    dot_label = "backend online" if health else "backend offline"
    st.markdown(
        f'<div class="status-line"><span class="status-dot {dot_class}"></span>'
        f'{dot_label} · {API_BASE_URL}</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# MAIN COMPOSER
# =========================================================================
def composer() -> None:
    """Prompt box with an inline model picker."""
    if not st.session_state.get("last_content"):
        st.markdown(
            '<div class="hero">'
            '<p class="hero-title">What should we write about?</p>'
            '<p class="hero-sub">Give a topic, pick a model, and the research-to-article pipeline handles the rest.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        topic = st.text_area(
            "Topic",
            label_visibility="collapsed",
            placeholder="e.g. How should production RAG systems be evaluated?",
            height=124,
            key="composer_topic",
        )
        cols = st.columns([1.5, 1, 1], vertical_alignment="center")
        options = ["Auto (fallback chain)"] + list(st.session_state.get("_model_options") or [])
        current = st.session_state.get("model_choice") or options[0]
        if current not in options:
            current = options[0]
        with cols[0]:
            with st.popover(f"Model · {current.replace('gemini/', '')}", use_container_width=True):
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
            submitted = st.button("Send to draft", type="primary", use_container_width=True)
    if submitted:
        run_generation((topic or "").strip(), as_of, current)


# Ordered pipeline steps shown as a live checklist. Keys must match the
# ``stage`` values written by app.services.jobs._stream_graph_to_completion,
# including the "queued" pre-start state and the "skipped_research" marker
# written when the router goes straight to the planner (closed-book topics).
PIPELINE_STEPS: list[tuple[str, str]] = [
    ("router", "Router"),
    ("research", "Research"),
    ("planner", "Planner"),
    ("writing", "Writer"),
    ("merging", "Merge"),
    ("quality_gate", "Quality gate"),
    ("revising", "Revise"),
    ("images", "Images"),
    ("finishing", "Final blog"),
]


def _friendly_stage(stage: str | None) -> str:
    name = (stage or "").lower()
    mapping = {
        "queued": "Queued", "router": "Router", "research": "Researcher",
        "skipped_research": "Researcher", "planner": "Planner",
        "writing": "Writer", "merging": "Merge",
        "quality_gate": "Quality gate", "revising": "Revise",
        "images": "Images", "finishing": "Final blog",
        "completed": "Final blog", "failed": "Failed",
        # Back-compat with stage values written by older worker versions.
        "run": "Writer", "workers": "Writer", "quality gate": "Quality gate",
    }
    return mapping.get(name, stage or PIPELINE_STEPS[0][1])


def _stage_fraction(stage: str | None, progress: float | None, attempt: int, total: int) -> float:
    """Resolve the bar position: backend fraction first, time fallback."""
    try:
        value = float(progress) if progress is not None else float("nan")
    except (TypeError, ValueError):
        value = float("nan")
    if value == value:  # not NaN -> backend reported a real fraction
        return max(0.0, min(1.0, value))
    order = [key for key, _ in PIPELINE_STEPS]
    name = (stage or "").lower()
    if name in order:
        return (order.index(name) + 1) / (len(order) + 1)
    return min(0.95, (attempt + 1) / max(total, 1) * 0.95)


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
    st.session_state.last_job_id = job_id
    progress = st.progress(0, text="Starting workflow...")
    status_box = st.status("Router — starting...", expanded=True)
    checklist_slot = st.empty()
    shown: set[str] = set()
    last_fraction = 0.0
    # Skipped-node rendering: the router may jump straight router -> planner
    # (closed-book topics skip research). A node that never ran shows "-" so
    # the checklist always mirrors the real path instead of pretending every
    # node ran. Backend writes the final article into the job row the moment
    # the graph finishes; the loop below polls every 2s and calls st.rerun()
    # the instant status flips to "completed" (~2s latency).
    skipped: set[str] = set()
    total_attempts = 300  # ~10 min at a 2s poll interval
    for attempt in range(total_attempts):
        if attempt:
            time.sleep(2)
        result = api_request("GET", f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        if not result.ok:
            progress.empty()
            show_error(result)
            return
        job = result.json()
        if job["status"] == "completed":
            order = [key for key, _ in PIPELINE_STEPS]
            lines = []
            for key, label in PIPELINE_STEPS:
                if key in ("revising",):
                    lines.append(f"— {label} (not needed)")
                else:
                    lines.append(f"✓ {label}")
            checklist_slot.markdown("\n\n".join(lines))
            st.session_state.last_content = job.get("content", "")
            st.session_state.last_job_id = job_id
            st.session_state.last_plan = job.get("plan")
            st.session_state.last_evidence = job.get("evidence", [])
            st.session_state.last_timestamps = (job.get("created_at"), job.get("updated_at"))
            progress.progress(1.0, text="Article ready")
            status_box.update(label="Done", state="complete", expanded=False)
            st.rerun()
        if job["status"] == "failed":
            progress.empty()
            status_box.update(label="Generation failed", state="error", expanded=True)
            st.error(job.get("error") or "Article generation failed.")
            return
        stage = (job.get("stage") or "router").lower()
        detail = job.get("stage_detail") or _friendly_stage(stage)
        # The router's "next" marker tells us research was skipped entirely:
        # Router -/Planner active instead of a fake Research tick.
        if stage == "skipped_research":
            skipped.add("research")
            stage = "planner"
            detail = "Research skipped (evergreen topic) — planning sections…"
        fraction = _stage_fraction(stage, job.get("progress"), attempt, total_attempts)
        # Never let the bar move backwards when a node retries.
        fraction = max(fraction, last_fraction)
        last_fraction = fraction
        friendly = _friendly_stage(stage)
        status_box.update(label=f"{friendly} — {detail}", expanded=True)
        if stage not in shown:
            status_box.write(f"Stage: `{stage}` — {detail}")
            shown.add(stage)
        order = [key for key, _ in PIPELINE_STEPS]
        current_idx = order.index(stage) if stage in order else -1
        lines = []
        for idx, (key, label) in enumerate(PIPELINE_STEPS):
            if key in skipped:
                lines.append(f"— {label} (skipped)")
            elif idx < current_idx:
                lines.append(f"✓ {label}")
            elif idx == current_idx:
                lines.append(f"● **{label}** — {detail}")
            else:
                lines.append(f"○ {label}")
        checklist_slot.markdown("\n\n".join(lines))
        progress.progress(fraction, text=f"{job['status']} · {friendly} · {detail}")
    else:
        progress.empty()
        st.warning(f"Still running. Job ID: {job_id}")


# =========================================================================
# ARTICLE VIEW — rendered on a "paper" card
# =========================================================================
def article_view() -> None:
    import html as html_mod

    markdown = st.session_state.get("last_content", "")
    if not markdown:
        return
    title = extract_title(markdown)
    evidence = st.session_state.get("last_evidence") or extract_evidence(markdown)
    image_count = len(IMAGE_RE.findall(markdown))
    word_count = len(re.findall(r"\w+", markdown))
    created, updated = st.session_state.get("last_timestamps", (None, None))

    st.markdown('<div class="paper">', unsafe_allow_html=True)
    st.markdown(f'<p class="article-title">{html_mod.escape(title)}</p>', unsafe_allow_html=True)
    byline = '<span class="status-pill">Completed</span>'
    byline += f'<span>job {str(st.session_state.get("last_job_id"))[:8]}</span>'
    if created:
        byline += f'<span>created {str(created)[:10]}</span>'
    if updated:
        byline += f'<span>updated {str(updated)[:10]}</span>'
    st.markdown(f'<div class="byline">{byline}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-line">'
        f'<div class="stat-block"><div class="stat-num">{word_count:,}</div><div class="stat-lbl">Words</div></div>'
        f'<div class="stat-block"><div class="stat-num">{len(evidence)}</div><div class="stat-lbl">Sources</div></div>'
        f'<div class="stat-block"><div class="stat-num">{image_count}</div><div class="stat-lbl">Images</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

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
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "Download Markdown", markdown.encode("utf-8"), f"{safe_slug(title)}.md",
                "text/markdown", use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "Download Bundle (MD + images)", bundle_bytes(markdown, title),
                f"{safe_slug(title)}_bundle.zip", "application/zip", use_container_width=True,
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
            f"Status: completed\nEvidence found: {len(evidence)}\n"
            f"Images found: {image_count}\n"
            f"Created: {created}\nUpdated: {updated}"
        )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# APP FLOW
# =========================================================================
if st.session_state.pop("signup_done", False):
    st.toast("Welcome! Your account is ready — you're signed in.", icon="🖋️")

try:
    health_response = api_request("GET", "/api/v1/health")
    health = health_response.json() if health_response.ok else None
except requests.RequestException:
    health = None

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