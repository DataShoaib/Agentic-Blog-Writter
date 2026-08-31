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
OUTPUTS_DIR = Path("outputs")
IMAGES_DIR = Path("images")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
# Polling budget mirrors APP_CONFIG.job_timeout_seconds on the backend.
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 900
POLL_MAX_ATTEMPTS = POLL_TIMEOUT_SECONDS // POLL_INTERVAL_SECONDS
GRAPH_STAGES = ["router", "research", "planner", "worker", "merge", "quality", "revise", "images", "generate_images"]

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
    return requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)


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


def load_blog(token: str, job_id: str) -> None:
    """Load a previously generated blog into the workspace from its job record."""
    try:
        response = api_request(
            "GET", f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return
    if not response.ok:
        show_error(response)
        return
    job = response.json()
    if job.get("status") != "completed":
        st.warning(f"Blog {job_id} is not completed (status: {job.get('status')}).")
        return
    st.session_state.last_content = job.get("content", "")
    st.session_state.last_job_id = job_id
    st.session_state.last_plan = job.get("plan")
    st.session_state.last_evidence = job.get("evidence", [])
    st.session_state.last_stage = job.get("stage", "completed")
    st.session_state.last_created_at = job.get("created_at")
    st.session_state.last_updated_at = job.get("updated_at")
    st.session_state.last_executor = "cached"
    st.rerun()


def signup_panel() -> None:
    with st.form("signup-form"):
        username = st.text_input("Username", placeholder="writer")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        try:
            response = api_request("POST", "/api/v1/auth/signup", json={"username": username, "password": password})
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {API_BASE_URL}: {exc}")
            return
        if response.ok:
            st.success("Account created. Open Login to continue.")
        else:
            show_error(response)


def login_panel() -> None:
    with st.form("login-form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
    if submitted:
        if not username.strip() or not password.strip():
            st.warning("Enter both a username and a password.")
            return
        try:
            response = api_request("POST", "/api/v1/auth/token", data={"username": username, "password": password})
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {API_BASE_URL}: {exc}")
            return
        if response.ok:
            st.session_state.token = response.json()["access_token"]
            st.rerun()
        else:
            show_error(response)


def authenticated_workspace() -> None:
    token = st.session_state.get("token")
    with st.sidebar:
        st.header("Generate New Blog")
        topic = st.text_area("Topic", placeholder="How should production RAG systems be evaluated?", height=120)
        research_label = st.radio(
            "Web research",
            ["Auto", "Always", "Never"],
            horizontal=True,
            help=(
                "Auto: the router decides (stable concepts like self-attention skip research). "
                "Always: force web search. Never: skip research entirely."
            ),
        )
        st.session_state["research_mode"] = {"Auto": "auto", "Always": "force", "Never": "skip"}[research_label]
        submitted = st.button("Generate Blog", type="primary", use_container_width=True)
        st.divider()

        # Per-user blog history: every blog this user generated, selectable.
        st.subheader("My past blogs")
        try:
            resp = api_request(
                "GET", "/api/v1/blogs", headers={"Authorization": f"Bearer {token}"}
            )
        except requests.RequestException:
            resp = None
        if resp is not None and resp.ok:
            blogs = resp.json().get("blogs", [])
            if not blogs:
                st.caption("No blogs generated yet.")
            else:
                labels = [
                    f"{b['title']}  ·  {b['job_id'][:8]}" for b in blogs
                ]
                chosen = st.radio(
                    "Select a blog to reload",
                    labels,
                    key="past_blog_radio",
                    label_visibility="collapsed",
                )
                if st.button("Load selected blog", use_container_width=True):
                    index = labels.index(chosen)
                    load_blog(token, blogs[index]["job_id"])
        else:
            st.caption("Could not load blog history.")

        st.divider()
        if st.button("Log out", use_container_width=True):
            st.session_state.pop("token", None)
            st.rerun()

    if submitted:
        if not topic.strip():
            st.warning("Please enter a topic.")
            return
        try:
            response = api_request(
                "POST",
                "/api/v1/generate",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "topic": topic.strip(),
                    "research_mode": st.session_state.get("research_mode", "auto"),
                },
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {API_BASE_URL}: {exc}")
            return
        if not response.ok:
            show_error(response)
            return
        job_id = response.json()["job_id"]
        executor = health.get("jobs_executor", "unknown") if health else "unknown"
        progress = st.progress(0, text=f"Job queued on the {executor} executor...")
        status_box = st.status("Running graph...", expanded=True)
        status_box.write(f"Job ID: `{job_id}` | Executor: `{executor}`")
        shown_stage: str | None = None
        started_at = time.time()
        poll_errors = 0
        for attempt in range(POLL_MAX_ATTEMPTS):
            time.sleep(POLL_INTERVAL_SECONDS)
            elapsed = int(time.time() - started_at)
            try:
                result = api_request(
                    "GET",
                    f"/api/v1/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
            except requests.RequestException as exc:
                poll_errors += 1
                if poll_errors >= 5:
                    progress.empty()
                    status_box.update(label="Lost connection to the backend", state="error", expanded=True)
                    st.error(f"Stopped polling job `{job_id}` after repeated network errors: {exc}")
                    return
                status_box.write(f"Polling hiccup ({poll_errors}/5): {exc}")
                continue
            if not result.ok:
                progress.empty()
                show_error(result)
                return
            poll_errors = 0
            job = result.json()
            stage = job.get("stage") or job["status"]
            if stage != shown_stage:
                status_box.write(f"Stage: `{stage}` — {elapsed}s elapsed")
                shown_stage = stage
            progress.progress(
                min(95, elapsed * 95 // POLL_TIMEOUT_SECONDS),
                text=f"Status: {job['status']} | Stage: {stage} | {elapsed}s elapsed",
            )
            if job["status"] == "completed":
                st.session_state.last_content = job.get("content", "")
                st.session_state.last_job_id = job_id
                st.session_state.last_plan = job.get("plan")
                st.session_state.last_evidence = job.get("evidence", [])
                st.session_state.last_stage = stage
                st.session_state.last_created_at = job.get("created_at")
                st.session_state.last_updated_at = job.get("updated_at")
                st.session_state.last_elapsed_seconds = elapsed
                st.session_state.last_executor = executor
                progress.progress(100, text=f"Article ready in {elapsed}s")
                status_box.update(label="Done", state="complete", expanded=False)
                break
            if job["status"] == "failed":
                progress.empty()
                status_box.update(label=f"Generation failed at stage `{stage}`", state="error", expanded=True)
                st.error(job.get("error") or "Article generation failed.")
                st.caption(f"Job ID for server logs: `{job_id}` | Executor: `{executor}`")
                return
        else:
            progress.empty()
            st.warning(
                f"The job is still running after {POLL_TIMEOUT_SECONDS}s. "
                f"It keeps executing on the server; check back later. Job ID: `{job_id}`"
            )

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
            f"API: {API_BASE_URL}\n"
            f"Job: {st.session_state.get('last_job_id')}\n"
            f"Executor: {st.session_state.get('last_executor', 'unknown')}\n"
            f"Final stage: {st.session_state.get('last_stage', 'completed')}\n"
            f"Created at: {st.session_state.get('last_created_at', 'n/a')}\n"
            f"Updated at: {st.session_state.get('last_updated_at', 'n/a')}\n"
            f"Wall time: {st.session_state.get('last_elapsed_seconds', '?')}s\n"
            f"LangSmith project: {health.get('langsmith_project') if health and health.get('langsmith_tracing') else 'tracing off'}\n"
            f"Evidence found: {len(st.session_state.get('last_evidence') or extract_evidence(markdown))}\n"
            f"Images found: {len(IMAGE_RE.findall(markdown))}"
        )


st.markdown('<div class="hero"><h1>Agentic Content Orchestrator</h1><p>Research, draft, review, and export your article.</p></div>', unsafe_allow_html=True)
try:
    health_response = api_request("GET", "/api/v1/health")
    health = health_response.json() if health_response.ok else None
except requests.RequestException:
    health = None

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    if health:
        st.success("Backend online")
        st.caption(f"Images enabled: {health.get('images_enabled', False)}")
        st.caption(f"Redis ready: {health.get('redis_ready', False)}")
        st.caption(f"Jobs executor: {health.get('jobs_executor', 'unknown')}")
        if health.get("langsmith_tracing"):
            st.caption(f"LangSmith: {health.get('langsmith_project')}")
        elif str(health.get("langsmith_auth", "")).startswith("LangSmithAuthError") or "401" in str(health.get("langsmith_auth", "")) or "Invalid token" in str(health.get("langsmith_auth", "")):
            st.error("LangSmith key invalid - traces NOT uploading. Fix LANGSMITH_API_KEY in .env and restart.")
        else:
            st.caption("LangSmith tracing: off")
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