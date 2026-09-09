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
OUTPUTS_DIR = Path("outputs")
IMAGES_DIR = Path("images")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

st.set_page_config(page_title="Agentic Content Orchestrator", page_icon="✦", layout="wide")
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
        color: #172033 !important;
    }
    [data-testid="stAppViewContainer"] { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dbe2ec; }
    [data-testid="stHeader"] { background: #f5f7fb; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-baseweb="select"] *, [data-baseweb="input"] * {
        color: #172033 !important; background: #ffffff !important;
    }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder {
        color: #718096 !important;
    }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        color: #172033 !important; background: #ffffff !important; border-color: #b8c4d4 !important;
    }
    [data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
        color: #ffffff !important; background: #0b63ce !important; border-color: #0b63ce !important;
    }
    [data-testid="stExpander"] details, [data-testid="stStatusWidget"] {
        background: #ffffff !important; border: 1px solid #dbe2ec !important;
    }
    [data-baseweb="tab-list"] { gap: 8px; }
    [data-baseweb="tab"] { color: #53657d !important; }
    [aria-selected="true"] { color: #0b63ce !important; border-bottom-color: #0b63ce !important; }
    [data-testid="stCode"] *, code, pre, [data-testid="stCodeBlock"] * {
        color: #172033 !important; background: #eef2f7 !important;
    }
    [data-testid="stMarkdownContainer"] a { color: #0b63ce !important; }
    [data-testid="stDataFrame"] * { color: #172033 !important; }
    .hero { padding: 1.5rem 0 .8rem; }
    .hero h1 { color: #10233f; font-size: 2.4rem; margin-bottom: .25rem; }
    .hero p { color: #53657d; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
            st.error("❌ Password must be at least 8 characters. You entered " + str(len(password)) + ".")
            return
        if not re.match(r"^[A-Za-z0-9_.-]+$", username):
            st.error("❌ Username may only contain letters, numbers, dots, dashes and underscores (no spaces or @).")
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
            st.session_state.pop("login_username", None)
            st.session_state.pop("login_password", None)
            st.rerun()
        else:
            st.session_state.login_username = login_username
            show_error(response)


def authenticated_workspace() -> None:
    token = st.session_state.get("token")
    if st.session_state.pop("signup_done", False):
        st.success("Welcome! Your account is ready — you're signed in.")
    with st.sidebar:
        st.header("Generate New Blog")
        topic = st.text_area("Topic", placeholder="How should production RAG systems be evaluated?", height=120)
        as_of = st.date_input("As-of date", value=None, help="Leave blank for today. Limits research to this date or earlier.")
        preferred_model = st.session_state.get("_model_options")
        model_choice = st.selectbox(
            "Preferred model",
            ["(fallback chain)"] + (preferred_model or []),
            index=0,
            help="Primary LLM for drafting/review. The fallback chain kicks in on quota errors.",
        )
        submitted = st.button("Generate Blog", type="primary", use_container_width=True)
        st.divider()

        st.subheader("My past blogs")
        try:
            history = api_request("GET", "/api/v1/blogs", headers={"Authorization": f"Bearer {token}"})
            blogs = history.json().get("blogs", []) if history.ok else []
        except Exception:
            blogs = []
        if not blogs:
            st.caption("No blogs generated yet.")
        else:
            labels = [f"{blog['title']} · {blog['created_at'][:10]}" for blog in blogs]
            choice = st.selectbox("Select a blog", labels, label_visibility="collapsed")
            if st.button("Load selected blog", use_container_width=True):
                selected = blogs[labels.index(choice)]
                result = api_request(
                    "GET",
                    f"/api/v1/jobs/{selected['job_id']}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if result.ok:
                    job = result.json()
                    st.session_state.last_content = job.get("content", "")
                    st.session_state.last_job_id = selected["job_id"]
                    st.session_state.last_plan = job.get("plan")
                    st.session_state.last_evidence = job.get("evidence", [])
                    st.rerun()
                else:
                    show_error(result)

        st.divider()
        if st.button("Log out", use_container_width=True):
            for key in ("token", "refresh_token", "last_content", "last_job_id", "last_plan", "last_evidence", "last_timestamps"):
                st.session_state.pop(key, None)
            st.rerun()

    if submitted:
        if not topic.strip():
            st.warning("Please enter a topic.")
            return
        payload = {
            "topic": topic.strip(),
            "preferred_model": None if model_choice == "(fallback chain)" else model_choice,
        }
        if as_of is not None:
            payload["as_of"] = as_of.isoformat()
        response = api_request("POST", "/api/v1/generate", headers={"Authorization": f"Bearer {token}"}, json=payload)
        if not response.ok:
            show_error(response)
            return
        job_id = response.json()["job_id"]
        progress = st.progress(0, text="Starting workflow...")
        status_box = st.status("Running graph...", expanded=True)
        stages = ["router", "research", "planner", "workers", "quality gate", "images"]
        shown_stages: set[str] = set()
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
                break
            if job["status"] == "failed":
                progress.empty()
                status_box.update(label="Generation failed", state="error", expanded=True)
                st.error(job.get("error") or "Article generation failed.")
                return
            # Real workflow stage reported by the backend (falls back to a heuristic).
            stage = job.get("stage") or stages[min(len(stages) - 1, attempt // 10)]
            if stage not in shown_stages:
                status_box.write(f"Stage: `{stage}`")
                shown_stages.add(stage)
            progress.progress(min(95, (attempt + 1) * 95 // 60), text=f"Job status: {job['status']} | {stage}")
        else:
            progress.empty()
            st.warning(f"The job is still running. Job ID: {job_id}")

    markdown = st.session_state.get("last_content", "")
    if not markdown:
        st.info("Enter a topic and click Generate Blog.")
        return

    title = extract_title(markdown)
    plan_tab, evidence_tab, preview_tab, images_tab, logs_tab = st.tabs(["Plan", "Evidence", "Markdown Preview", "Images", "Logs"])
    with plan_tab:
        st.subheader("Plan")
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
        st.subheader("Evidence")
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
        st.subheader("Markdown Preview")
        render_markdown(markdown)
        st.download_button("Download Markdown", markdown.encode("utf-8"), f"{safe_slug(title)}.md", "text/markdown")
        st.download_button("Download Bundle (MD + images)", bundle_bytes(markdown, title), f"{safe_slug(title)}_bundle.zip", "application/zip")
    with images_tab:
        st.subheader("Images")
        sources = [image_url(match.group("src")) for match in IMAGE_RE.finditer(markdown)]
        if not sources:
            st.info("No images were generated for this article.")
        for source in sources:
            st.image(source, use_container_width=True)
    with logs_tab:
        st.subheader("Logs")
        st.code(
            f"API: {API_BASE_URL}\nJob: {st.session_state.get('last_job_id')}\n"
            f"Status: completed\nEvidence found: {len(st.session_state.get('last_evidence') or extract_evidence(markdown))}\n"
            f"Images found: {len(IMAGE_RE.findall(markdown))}\n"
            f"Created: {st.session_state.get('last_timestamps', (None, None))[0]}\n"
            f"Updated: {st.session_state.get('last_timestamps', (None, None))[1]}"
        )


st.markdown('<div class="hero"><h1>Agentic Content Orchestrator</h1><p>Research, draft, review, and export your article.</p></div>', unsafe_allow_html=True)
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

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    if health:
        st.success("Backend online")
        st.caption(f"JWT configured: {health.get('jwt_configured', False)}")
        st.caption(f"Redis ready: {health.get('redis_ready', False)}")
        st.caption(f"Images enabled: {health.get('images_enabled', False)}")
        if st.session_state._model_options:
            st.caption(f"Models: {', '.join(st.session_state._model_options[:2])}")
    else:
        st.error("Backend offline")

if st.session_state.get("token"):
    authenticated_workspace()
else:
    signup_tab, login_tab = st.tabs(["Create account", "Login"])
    with signup_tab:
        signup_panel()
    with login_tab:
        login_panel()