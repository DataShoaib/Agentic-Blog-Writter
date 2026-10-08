from __future__ import annotations

import html as html_mod
import io
import os
import re
import time
import zipfile
from pathlib import Path

import config

# Force Streamlit's *native* theme to dark before the module initializes.
# Several built-in widgets (st.dataframe's grid, st.progress, st.json,
# radio/checkbox/toggle controls) are rendered by Streamlit itself using
# its configured theme colors, not by our custom CSS below — if the native
# theme stays on its "light" default, those widgets render as pale/white
# blocks no matter what CSS is injected afterwards. Setting the theme via
# environment variables (read once, at Streamlit's startup) keeps every
# native control consistent with the "Slate & Signal" palette used
# everywhere else in this file.
for _env_key, _env_value in config.STREAMLIT_THEME_ENV.items():
    os.environ.setdefault(_env_key, _env_value)

import requests
import streamlit as st

API_BASE_URL = config.API_BASE_URL
IMAGES_DIR = config.IMAGES_DIR
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="centered")

# =========================================================================
# DESIGN — "Slate & Signal"
# A dark, flat SaaS/enterprise look: dark slate app chrome, a single blue
# accent (no gradients, no warm/paper textures), sans-serif everywhere
# (Inter). The generated article renders on a plain white card — clean and
# readable, like a document inside a product, not a manuscript on a desk.
# Icons are plain text/characters, not emoji or font-ligatures, so
# there is no font-load path that can turn an icon into stray text.
# =========================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

/* ---------- app shell: dark slate ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 480px at 85% -10%, rgba(61,139,253,.07), transparent 60%),
        radial-gradient(700px 420px at 0% 100%, rgba(61,139,253,.04), transparent 55%),
        #0d0f14;
}
[data-testid="stHeader"] {
    background: rgba(13,15,20,.85) !important;
    border-bottom: 1px solid rgba(255,255,255,.06) !important;
}
[data-testid="stHeader"] * { color: #e7e9f0 !important; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Streamlit text contrast — keep every native control readable ---------- */
[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] [data-testid="stAlert"] * {
    color: #dde0ea !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"] {
    background: #171b25 !important;
    border: 1px solid #2b3142 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"] p,
[data-testid="stAppViewContainer"] [data-testid="stAlert"] span,
[data-testid="stAppViewContainer"] [data-testid="stAlert"] div {
    color: #eef0f6 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stStatusWidget"],
[data-testid="stAppViewContainer"] [data-testid="stToast"] {
    color: #eef0f6 !important;
}

.block-container {
    padding-top: 2.3rem !important;
    padding-bottom: 4rem !important;
    max-width: 860px !important;
}
.block-container > div > [data-testid="stVerticalBlock"] { width: 100%; }
[data-testid="stVerticalBlock"] { row-gap: .85rem; }
[data-testid="stHorizontalBlock"] { column-gap: .7rem; align-items: stretch; }

h1,h2,h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family: 'Inter', sans-serif; font-weight: 700; }

/* ---------- sidebar: darkest layer ---------- */
[data-testid="stSidebar"] {
    background: #0b0d12 !important;
    border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: #aeb4c9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.07) !important; margin: .85rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .05rem; }

.brand { display: flex; align-items: center; gap: .65rem; padding: .15rem .1rem .8rem; }
.brand-mark {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.05rem;
    background: linear-gradient(160deg, #1b2431, #0f1218);
    border: 1px solid rgba(61,139,253,.4);
    color: #6ba6ff !important;
}
.brand-name { font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 700; color: #eef0f6 !important; line-height: 1.15; }
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
    background: rgba(255,255,255,.05) !important; color: #eef0f6 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #3d8bfd !important;
    color: #06111f !important; justify-content: center !important; font-weight: 700 !important;
    border-radius: 9px !important; padding: .55rem .7rem !important; font-size: .86rem !important;
    box-shadow: 0 4px 14px rgba(61,139,253,.22) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover { background: #5b9dfd !important; }

.hist-empty { font-size: .8rem; color: #5c6280; line-height: 1.55; padding: .3rem .1rem .5rem; }

.user-chip {
    display: flex; align-items: center; gap: .55rem;
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; padding: .5rem .65rem; font-size: .83rem; color: #cfd3e6 !important;
}
.user-chip .avatar {
    width: 22px; height: 22px; border-radius: 50%; background: #3d8bfd; color: #06111f !important;
    display: flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 700; flex-shrink: 0;
}
.status-line { display: flex; align-items: center; gap: .45rem; font-size: .72rem !important; color: #565c78 !important; margin-top: .55rem; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.on  { background: #5cc98a; box-shadow: 0 0 0 3px rgba(92,201,138,.16); }
.status-dot.off { background: #e15b5b; box-shadow: 0 0 0 3px rgba(225,91,91,.16); }

/* ---------- main canvas cards: dark ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #12151c;
    border: 1px solid #232838 !important;
    border-radius: 14px;
}

/* ---------- hero / composer copy ---------- */
.hero { margin: .2rem 0 1.15rem; }
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.85rem !important; font-weight: 700 !important; color: #eef0f6 !important;
    margin: 0 !important; line-height: 1.25 !important;
}
.hero-sub {
    color: #b7bdd0 !important;
    font-size: .96rem !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    margin: .38rem 0 0 !important;
    opacity: 1 !important;
}

/* ---------- composer (chat-input styled) ---------- */
[data-testid="stTextArea"] textarea {
    border-radius: 10px !important; border: 1.5px solid #232838 !important;
    background: #0f1218 !important; color: #e7e9f0 !important; font-size: .96rem !important;
    padding: .85rem .95rem !important; transition: border .15s ease, box-shadow .15s ease;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #3d8bfd !important; box-shadow: 0 0 0 3px rgba(61,139,253,.16) !important; outline: none;
}
[data-testid="stTextArea"] textarea::placeholder { color: #565c78 !important; }

[data-testid="stTextInput"] input {
    border-radius: 8px !important; border: 1.5px solid #232838 !important;
    background: #0f1218 !important; color: #e7e9f0 !important; font-size: .95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3d8bfd !important; box-shadow: 0 0 0 3px rgba(61,139,253,.15) !important;
}
[data-testid="stTextInput"] label, [data-testid="stDateInput"] label {
    font-weight: 600 !important; color: #9aa0bd !important; font-size: .8rem;
}
[data-testid="stDateInput"] input {
    border-radius: 8px !important; border: 1.5px solid #232838 !important;
    background: #0f1218 !important; color: #e7e9f0 !important; font-size: .85rem !important;
}

[data-testid="stPopover"] button {
    border-radius: 999px !important; border: 1.5px solid #232838 !important;
    background: #0f1218 !important; color: #cfd3e6 !important;
    font-size: .83rem !important; font-weight: 600; padding: .42rem .9rem !important;
    box-shadow: none !important; transition: border .15s ease;
}
[data-testid="stPopover"] button:hover { border-color: #3d8bfd !important; }
[data-testid="stPopoverBody"] { background: #12151c !important; border: 1px solid #232838 !important; }
[data-testid="stPopoverBody"] * { color: #cfd3e6 !important; }

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {
    background: #3d8bfd !important;
    color: #06111f !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; box-shadow: 0 4px 12px rgba(61,139,253,.18) !important;
    transition: background .15s ease;
}
[data-testid="stBaseButton-primary"]:hover { background: #5b9dfd !important; }
[data-testid="stBaseButton-secondary"] {
    border-radius: 8px !important; background: #0f1218 !important;
    border: 1.5px solid #232838 !important; color: #cfd3e6 !important; font-weight: 500;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #3d8bfd !important; color: #eef0f6 !important; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; background: #0f1218; padding: 4px; border-radius: 10px; border: 1px solid #232838;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 7px !important; color: #7d84a3 !important; font-weight: 600 !important;
    font-size: .86rem !important; background: transparent; border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #232838 !important; color: #eef0f6 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] p { color: #cfd3e6 !important; }

/* ---------- the article card: clean white surface, flat SaaS style ---------- */
.paper {
    background: #ffffff;
    border: 1px solid #e3e6ec;
    border-radius: 12px;
    padding: 1.7rem 1.9rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.3);
    margin-bottom: 1rem;
}
.paper ::selection {
    background: rgba(61,139,253,.22);
    color: #14161f;
}
.paper .article-title {
    font-family: 'Inter', sans-serif; font-size: 1.5rem !important; font-weight: 700 !important;
    color: #14161f !important; letter-spacing: -.01em; margin: 0 !important; line-height: 1.3 !important;
}
.paper .byline {
    font-size: .78rem !important; color: #6b7180 !important; margin: .4rem 0 0 !important;
    display: flex; flex-direction: column; align-items: flex-start; gap: .3rem !important;
}
.paper .byline .status-pill {
    background: #e4efe4; color: #35703f !important; border-radius: 999px;
    padding: .15rem .62rem; font-size: .72rem !important; font-weight: 700;
    align-self: flex-start;
}
.paper .byline .meta-row {
    display: flex; gap: .5rem; align-items: baseline;
}
.paper .byline .meta-label {
    font-weight: 700; color: #6b7180 !important;
}
.paper .byline .meta-value {
    color: #14161f !important;
}
.paper .stat-line {
    display: flex; flex-wrap: nowrap; margin: 1rem 0 .2rem;
    border-top: 1px solid #e3e6ec; border-bottom: 1px solid #e3e6ec;
}
.paper .stat-block {
    flex: 1 1 0; min-width: 0; padding: .55rem .25rem; text-align: center;
    border-right: 1px solid #e3e6ec; overflow-wrap: anywhere;
}
.paper .stat-block:last-child { border-right: none; }
.paper .stat-num { font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700; color: #14161f; }
.paper .stat-lbl { font-size: .68rem; color: #6b7180; font-weight: 600; }
.paper table { width: 100%; border-collapse: collapse; margin: .6rem 0; font-size: .85rem; }
.paper th, .paper td { border: 1px solid #e3e6ec; padding: .45rem .6rem; text-align: left; }
.paper th { background: #f3f5f8; }
.paper pre { max-width: 100%; overflow-x: auto; }
.paper [data-testid="stMarkdownContainer"] p,
.paper [data-testid="stMarkdownContainer"] li { color: #14161f !important; font-size: .98rem; line-height: 1.7; }
.paper [data-testid="stMarkdownContainer"] h1,
.paper [data-testid="stMarkdownContainer"] h2,
.paper [data-testid="stMarkdownContainer"] h3 { color: #14161f !important; }
.paper [data-testid="stMarkdownContainer"] a { color: #3d8bfd; }
.paper [data-testid="stDownloadButton"] button {
    background: #14161f !important; color: #ffffff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.paper [data-testid="stDownloadButton"] button:hover { background: #232838 !important; }

/* ---------- auth ---------- */
.auth-kicker { font-size: .76rem !important; font-weight: 600 !important; color: #3d8bfd !important; text-align: center !important; margin-bottom: .6rem !important; }
.auth-mark {
    width: 42px; height: 42px; border-radius: 12px; margin: .2rem auto .7rem;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    background: linear-gradient(160deg, #1b2431, #0f1218); border: 1px solid rgba(61,139,253,.4);
}
.auth-title { font-family: 'Inter', sans-serif; font-size: 1.35rem !important; font-weight: 700 !important; color: #eef0f6 !important; margin: 0 0 .15rem !important; text-align: center !important; }
.auth-sub { color: #7d84a3 !important; font-size: .89rem !important; margin: 0 0 1rem !important; text-align: center !important; line-height: 1.5 !important; }

/* ---------- misc ---------- */
[data-testid="stMarkdownContainer"] a { color: #3d8bfd; }
[data-testid="stExpander"] details { border-radius: 10px !important; border: 1px solid #232838 !important; background: #0f1218; }
[data-testid="stExpander"] summary p { color: #cfd3e6 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stCaptionContainer"] {
    color: #aeb5ca !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #aeb5ca !important;
}

/* Native Streamlit widgets: explicit contrast prevents invisible text across themes */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label,
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label,
[data-testid="stDateInput"] label {
    color: #cfd3e6 !important;
}
[data-baseweb="select"] *,
[data-baseweb="popover"] *,
[data-testid="stPopover"] * {
    color: #e5e8f0 !important;
}
[data-baseweb="select"] [role="option"] {
    background: #12151c !important;
    color: #e5e8f0 !important;
}
[data-baseweb="select"] [role="option"]:hover,
[data-baseweb="select"] [aria-selected="true"] {
    background: #232838 !important;
    color: #ffffff !important;
}
[data-testid="stProgress"] * {
    color: #e5e8f0 !important;
}

/* ---------- progress bar: strong contrast track + fill ---------- */
[data-testid="stProgress"] > div > div {
    background-color: #171b25 !important;
    border: 1px solid #2b3142 !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div > div > div {
    background-color: #3d8bfd !important;
    background-image: none !important;
    border-radius: 999px !important;
}

/* ---------- dataframes / tables (Evidence, Plan tasks) ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #0f1218 !important;
    border: 1px solid #232838 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] *, [data-testid="stTable"] * {
    color: #e5e8f0 !important;
}
[data-testid="stTable"] table { background: #0f1218 !important; }
[data-testid="stTable"] thead th {
    background: #171b25 !important; color: #cfd3e6 !important;
    border-bottom: 1px solid #2b3142 !important; font-weight: 700 !important;
}
[data-testid="stTable"] tbody td {
    border-bottom: 1px solid #1c202c !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) td { background: #12151c !important; }
[data-testid="stTable"] tbody tr:nth-child(odd) td  { background: #0f1218 !important; }

/* ---------- st.json viewer ---------- */
[data-testid="stJson"] {
    background: #0f1218 !important;
    border: 1px solid #232838 !important;
    border-radius: 10px !important;
}
[data-testid="stJson"] * { color: #e5e8f0 !important; }

/* ---------- code blocks ---------- */
[data-testid="stCodeBlock"], [data-testid="stCodeBlock"] pre {
    background: #0f1218 !important;
    border: 1px solid #232838 !important;
    border-radius: 10px !important;
}
[data-testid="stCodeBlock"] code { color: #e5e8f0 !important; }

/* ---------- radio / checkbox / toggle option text ---------- */
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p,
[data-testid="stToggle"] label p {
    color: #dfe3ef !important;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# =========================================================================
# BACKEND HELPERS — unchanged contract with your FastAPI service
# =========================================================================
def api_request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(
        method, f"{API_BASE_URL}{path}", timeout=config.REQUEST_TIMEOUT_SECONDS, **kwargs
    )
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
            response = requests.request(
                method, f"{API_BASE_URL}{path}", timeout=config.REQUEST_TIMEOUT_SECONDS, **kwargs
            )
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
    """Centered auth card, matching the Slate & Signal theme."""
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

            # A raw <div> opened in one st.markdown() call and closed in a
            # later one does NOT actually wrap whatever widget is rendered
            # in between — each st.markdown()/st.button() call is parsed by
            # Streamlit as its own standalone HTML fragment, so a wrapper
            # div spanning multiple calls is auto-closed immediately and
            # any CSS rule targeting it never matches anything real. A
            # keyed st.container() gives Streamlit's own "st-key-<key>"
            # class to the actual wrapping element, which a scoped
            # <style> block can target reliably for just the active item.
            safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", str(blog["job_id"]))
            with st.container(key=f"hist_{safe_key}"):
                clicked = st.button(
                    label,
                    key=f"hist_btn_{safe_key}",
                    use_container_width=True,
                    help="Open this article",
                )
            if blog["job_id"] == active_job:
                st.markdown(
                    f"<style>.st-key-hist_{safe_key} button {{"
                    "background: rgba(61,139,253,.12) !important;"
                    "border-left: 2px solid #3d8bfd !important;"
                    "color: #eef0f6 !important;"
                    "}}</style>",
                    unsafe_allow_html=True,
                )

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
        cols = st.columns([1.5, 1], vertical_alignment="center")
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
            submitted = st.button("Send to draft", type="primary", use_container_width=True)
        images_on = st.toggle(
            "Images",
            value=False,
            help="Off by default. Turn on to generate diagrams — you will be asked for a Gemini (Google AI Studio) API key.",
        )
        image_api_key = None
        if images_on:
            image_api_key = (
                st.text_input(
                    "Image API key (Gemini / Google AI Studio — needed only when Images is on)",
                    type="password",
                    placeholder="Paste your Google AI Studio (Gemini) API key",
                    key="image_api_key",
                ).strip()
                or None
            )
    if submitted:
        if images_on and not image_api_key:
            st.warning(
                "Images is on but no Gemini API key was entered — the article "
                "will be generated without images. Add a Google AI Studio "
                "(Gemini) API key, or switch Images off."
            )
        run_generation((topic or "").strip(), current, images_on, image_api_key)


def _friendly_stage(stage: str | None) -> str:
    return config.STAGE_LABELS.get((stage or "").lower(), stage or config.PIPELINE_STEPS[0][1])


def run_generation(
    topic: str,
    model_choice: str,
    enable_images: bool = False,
    image_api_key: str | None = None,
) -> None:
    """Submit the job, then poll until the article is ready.

    Each poll reads the live ``stage`` from the backend and updates one
    progress bar + a node checklist. The instant status flips to
    ``completed`` the article is stored in session state and the page
    reruns, so the finished blog appears right away (within one poll).
    The loop deliberately polls for up to config.POLL_MAX_ATTEMPTS seconds:
    a full run (research, four sections, quality gate, revision, images)
    can easily exceed ten minutes on free-tier quotas, and giving up early
    is what left the UI stuck on "Generating" while the backend had already
    finished.
    """
    token = st.session_state.get("token")
    if not topic:
        st.warning("Please enter a topic first.")
        return
    payload = {
        "topic": topic,
        "preferred_model": None if model_choice.startswith("Auto") else model_choice,
        "enable_images": enable_images,
        "image_api_key": image_api_key,
    }
    response = api_request("POST", "/api/v1/generate", headers={"Authorization": f"Bearer {token}"}, json=payload)
    if not response.ok:
        show_error(response)
        return
    job_id = response.json()["job_id"]
    st.session_state.last_job_id = job_id
    progress = st.progress(0.0, text="Starting workflow…")
    checklist_slot = st.empty()
    skipped: set[str] = set()
    last_fraction = 0.0
    for attempt in range(config.POLL_MAX_ATTEMPTS):
        if attempt:
            time.sleep(config.POLL_INTERVAL_SECONDS)
        result = api_request("GET", f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        if not result.ok:
            progress.empty()
            show_error(result)
            return
        job = result.json()

        # ── Done: store the article and rerun so the page renders it. ──
        if job["status"] == "completed":
            st.session_state.last_content = job.get("content", "")
            st.session_state.last_job_id = job_id
            st.session_state.last_plan = job.get("plan")
            st.session_state.last_evidence = job.get("evidence", [])
            st.session_state.last_timestamps = (job.get("created_at"), job.get("updated_at"))
            done_lines = "\n\n".join(
                f"— {label} (not needed)" if key == "revising" else f"✓ {label}"
                for key, label in config.PIPELINE_STEPS
            )
            checklist_slot.markdown(f"**Done — article ready**\n\n{done_lines}")
            progress.progress(1.0, text="Article ready")
            st.rerun()

        # ── Failed: show the error instead of spinning forever. ──
        if job["status"] == "failed":
            fail_lines = "\n\n".join(f"✗ {label}" for _, label in config.PIPELINE_STEPS)
            checklist_slot.markdown(f"**Generation failed**\n\n{fail_lines}")
            progress.empty()
            st.error(job.get("error") or "Article generation failed.")
            return

        # ── Still running: resolve the current node and move the bar. ──
        stage = (job.get("stage") or "router").lower()
        detail = job.get("stage_detail") or _friendly_stage(stage)
        if stage == "skipped_research":
            skipped.add("research")
            stage = "planner"
            detail = "Research skipped (evergreen topic) — planning sections…"
        # Prefer the backend's reported progress; fall back to stage order.
        raw = job.get("progress")
        try:
            fraction = float(raw) if raw is not None else float("nan")
        except (TypeError, ValueError):
            fraction = float("nan")
        if fraction != fraction:  # NaN → backend gave no fraction
            order = [key for key, _ in config.PIPELINE_STEPS]
            fraction = (order.index(stage) + 1) / (len(order) + 1) if stage in order else 0.05
        fraction = max(fraction, last_fraction)  # never go backwards
        last_fraction = fraction
        progress.progress(min(fraction, 1.0), text=f"{detail}")

        order = [key for key, _ in config.PIPELINE_STEPS]
        current_idx = order.index(stage) if stage in order else -1
        lines = []
        for idx, (key, label) in enumerate(config.PIPELINE_STEPS):
            if key in skipped:
                lines.append(f"— {label} (skipped)")
            elif idx < current_idx:
                lines.append(f"✓ {label}")
            elif idx == current_idx:
                lines.append(f"● **{label}** — {detail}")
            else:
                lines.append(f"○ {label}")
        checklist_slot.markdown("**Generating…**\n\n" + "\n\n".join(lines))
    else:
        progress.empty()
        st.warning(f"Still running. Job ID: {job_id}")

# =========================================================================
# ARTICLE VIEW — rendered on a clean white card
# =========================================================================
def article_view() -> None:
    markdown = st.session_state.get("last_content", "")
    if not markdown:
        return
    title = extract_title(markdown)
    evidence = st.session_state.get("last_evidence") or extract_evidence(markdown)
    image_count = len(IMAGE_RE.findall(markdown))
    word_count = len(re.findall(r"\w+", markdown))
    created, updated = st.session_state.get("last_timestamps", (None, None))

    byline = '<span class="status-pill">Completed</span>'
    byline += (
        '<div class="meta-row"><span class="meta-label">Job:</span>'
        f'<span class="meta-value">{html_mod.escape(str(st.session_state.get("last_job_id"))[:8])}</span></div>'
    )
    if created:
        byline += (
            '<div class="meta-row"><span class="meta-label">Created:</span>'
            f'<span class="meta-value">{html_mod.escape(str(created)[:10])}</span></div>'
        )
    if updated:
        byline += (
            '<div class="meta-row"><span class="meta-label">Updated:</span>'
            f'<span class="meta-value">{html_mod.escape(str(updated)[:10])}</span></div>'
        )

    # Title + byline + stat row are emitted in ONE st.markdown() call so
    # they stay truly nested inside .paper — Streamlit parses each
    # st.markdown() call as its own standalone HTML fragment and normalizes
    # (auto-closes) unclosed tags at the end of that fragment, so splitting
    # this across multiple calls silently breaks the layout.
    st.markdown(
        f"""
        <div class="paper">
            <div class="article-title">{html_mod.escape(title)}</div>
            <div class="byline">{byline}</div>
            <div class="stat-line">
                <div class="stat-block">
                    <div class="stat-num">{word_count:,}</div>
                    <div class="stat-lbl">Words</div>
                </div>
                <div class="stat-block">
                    <div class="stat-num">{len(evidence)}</div>
                    <div class="stat-lbl">Sources</div>
                </div>
                <div class="stat-block">
                    <div class="stat-num">{image_count}</div>
                    <div class="stat-lbl">Images</div>
                </div>
            </div>
        </div>
        """,
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