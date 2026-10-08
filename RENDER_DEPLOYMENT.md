# Render Deployment Guide — Agentic Blog Writer

Deploy the complete project (FastAPI API + RQ worker + Postgres + Redis) on
Render **without changing any application logic**. All deployment-only changes
are: `render.yaml` (new), `Dockerfile` (`$PORT`), `app/main.py` (env CORS),
`scripts/render_init_db.py` (new), `.env.example` (new `CORS_ORIGINS`).

> **Read first — Render limits that shape this deployment**
>
> - **Background Workers have no free tier** → the worker needs the cheapest
>   paid plan (Starter). This is the one hard cost of this deployment.
> - **Free Postgres**: 1 GB, **expires after 30 days**, no backups.
> - **Free Key Value**: **in-memory only** — queue/cache/rate-limit data is
>   lost on every restart. All three are reconstructible, so this is safe,
>   but a deploy/restart means re-running in-flight jobs.
> - Free web services **spin down on idle** (first request after idle is slow).
>
> For a demo/evaluation deploy the free Postgres + free Key Value + Starter
> worker combination works. For anything real, upgrade Postgres and Key Value.

## 1. Create / deploy the Render Blueprint

1. Push this repo to GitHub.
2. In the Render Dashboard → **New → Blueprint** → select the repo.
3. Render reads `render.yaml` and proposes 4 resources:
   `blogwriter-api` (Web Service), `blogwriter-worker` (Background Worker),
   `blogwriter-db` (PostgreSQL), `blogwriter-cache` (Key Value).
4. Fill the prompted secrets (see §6), pick the **worker plan = Starter**
   (paid — no free tier exists), keep the rest on free plans, click **Apply**.
5. The API runs `preDeployCommand: python scripts/render_init_db.py`
   automatically before the first deploy goes live (see §7).

## 2. Create / configure PostgreSQL

The Blueprint creates `blogwriter-db` (database `blogwriter`, user
`blogwriter`, generated password — **never hardcoded anywhere**). The API and
worker both receive it as `DATABASE_URL` via `fromDatabase … property:
connectionString`. Nothing to configure by hand except the plan:
free (with the 30-day/limits caveat above) or paid for real use.

## 3. Create / configure Redis

The Blueprint creates `blogwriter-cache` as **Render Key Value**
(Redis-compatible, Valkey 8) with `ipAllowList: []` (private network only)
and `maxmemoryPolicy: allkeys-lru`. Both API and worker receive the **same**
instance as `REDIS_URL` via `fromService … property: connectionString`
(internal `redis://red-…:6379`, same region). Nothing else to configure.
Free tier keeps data in memory only — see §15.

## 4. Configure the API Web Service

Blueprint defaults suffice. Key points:

- `runtime: docker`, `Dockerfile` CMD expands Render's `$PORT` at container
  start (`--port ${PORT:-8000}`), `0.0.0.0`. No hardcoded 8000 in production.
- `healthCheckPath: /api/v1/health` — Render uses it for deploy health and
  uptime probes. Expected: `{"status": "ok"|"degraded", ...}`.
- If no Redis is reachable the API/routes still start; the rate limiter fails
  open and job submission falls back to the in-process thread executor — the
  `status` field reports `degraded` instead of refusing to boot (logs show
  the reason). This is intentional for Render cold starts, where Postgres or
  Key Value may not be reachable for the first seconds.

## 5. Configure the Background Worker

Blueprint `dockerCommand: python -m app.worker` — the unchanged entrypoint.
Same image, same env (`DATABASE_URL`, `REDIS_URL`, LLM keys, JWT key).
The worker **must share the API's secrets** — it signs nothing, but it reads
jobs/users and its LangGraph run uses the same LLM keys. If the worker is
stopped, the API keeps working via its thread-executor fallback (jobs die
with the process — run the worker in production).

## 6. Add all required environment variables

`DATABASE_URL` and `REDIS_URL` are auto-wired by the Blueprint. Everything
below is entered **manually in the Dashboard** (Blueprint declares them with
`sync: false` so Render prompts for them; same values on BOTH services):

| Variable | Required? | Notes |
|---|---|---|
| `GEMINI_API_KEY` | **yes** | Primary LLM + image provider. [Google AI Studio](https://aistudio.google.com) |
| `GROQ_API_KEY` | yes (fallback) | Needed for the `groq/*` fallback routes (separate free quota) |
| `TAVILY_API_KEY` | yes (research) | Without it, research returns 0 results and all topics degrade |
| `GOOGLE_API_KEY` | no | Alias supported by the code; leave empty if unused |
| `JWT_SECRET_KEY` | **yes** | 32+ random bytes (`secrets.token_urlsafe(48)`). Rotating logs everyone out |
| `LLM_MODEL` | no | Defaults to `gemini/gemini-2.5-flash` |
| `LLM_FALLBACK_MODELS` | no | JSON array; defaults cover Gemini-lite + Groq |
| `CORS_ORIGINS` | **yes on Render** | Comma-separated browser origins, e.g. `https://<frontend>.onrender.com`. Leave empty locally |
| `LANGSMITH_API_KEY` | no | Enables LangSmith tracing; omit to stay fully offline |
| `LOG_LEVEL` | no | Defaults `INFO` |

Never commit real values — `.env.example` is the complete template.


## 7. Run database migrations / initialization

There is **no migration framework** — the app creates its own tables lazily
on first use (`JobStore`/`UserStore` `CREATE TABLE IF NOT EXISTS` +
`ADD COLUMN IF NOT EXISTS`; `PostgresSaver.setup()` for checkpoints) and
every statement is idempotent. The Blueprint additionally runs
`preDeployCommand: python scripts/render_init_db.py` before each deploy,
which executes exactly those same creators and blocks the deploy (non-zero
exit) only when the database is unreachable. Nothing destructive is ever run.

## 8. Test `/health`

```bash
curl https://<api>.onrender.com/api/v1/health
# {"status":"ok","authentication":"required","redis_configured":true,
#  "redis_ready":true,"jwt_configured":true,"images_enabled":true}
```

`status` is `ok` only when Redis is reachable **and** a JWT secret is set;
otherwise `degraded` with the individual flags telling you which piece is
missing (see §4 for the fail-open rationale).

## 9. Test the API

```bash
curl -X POST https://<api>.onrender.com/api/v1/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","password":"secret-password"}'   # → 201
TOKEN=$(curl -s -X POST https://<api>.onrender.com/api/v1/auth/token \
  -d 'username=demo&password=secret-password' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl https://<api>.onrender.com/api/v1/models
# → the 7-model fallback chain
```

## 10. Test background job creation

```bash
curl -X POST https://<api>.onrender.com/api/v1/generate \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"topic":"How does self-attention work in transformers?"}'
# → 202 {"job_id":"...","status":"queued"}
```

A repeated identical request on the same day returns the Redis-cached result
instantly (key includes topic + date + images flag).

## 11. Test worker processing

```bash
curl https://<api>.onrender.com/api/v1/jobs/<job_id> \
  -H "Authorization: Bearer $TOKEN"
# status moves queued → running (with live "stage"/"progress") → completed
```

Watch the worker's logs in the Dashboard for the `router → research →
planner → writing → merge → quality → images` stages. Stuck at `queued` =
worker not running or wrong `REDIS_URL`. Stuck at `running` > 15 min =
worker was killed (RQ 900 s timeout is recorded on the row by the
`on_failure` hook; a row still `running` after a restart is reaped by the
owner-scoped sweep).


## 12. Test generated blog output

```bash
curl https://<api>.onrender.com/api/v1/jobs/<job_id> \
  -H "Authorization: Bearer $TOKEN" | python -c 'import sys,json;print(json.load(sys.stdin)["content"][:500])'
# → full Markdown article; served from the Postgres job row
```

The article body, plan and evidence live in the `jobs` table — listing
`/api/v1/blogs` returns the user's history.

## 13. Test generated images

Regenerate with `"enable_images": true` and a per-job
`"image_api_key"` (Google AI Studio key): the returned Markdown contains
`![…](/assets/images/<job_id>/…)` links and the API serves the bytes. If the
key is missing/quota-dead, the article ships with a visible note instead of
failing (§14 explains what survives).

## 14. What happens to `outputs/` and `images/` on Render

| Path | Written by | Read by? | On Render |
|---|---|---|---|
| `outputs/<job>.md` | worker, after every run | **nobody** — articles are read back from the Postgres `jobs` row, never from this file | Best-effort convenience copy; survives only until redeploy/restart. Harmless. |
| `images/<job>/*.png` | worker, during image node | **yes** — the API's `StaticFiles` mount `/assets/images` + the frontend's Images tab link to these bytes | **Ephemeral.** Render's filesystem is per-instance and wiped on every deploy/restart. Image markdown survives (Postgres), but the PNG bytes behind `/assets/images/…` links return 404 after a restart/deploy — the code never re-checks, so expect broken image links (article text unaffected). |

Important related limit: the API and worker are **separate containers with
separate filesystems**, so even without a restart the API cannot see PNGs
generated by the worker (this affects the `/assets/images` serving path in
`docker-compose.yml` too unless volumes are shared — Compose shares them via
`volumes:`, Render does not).

## 15. Storage limitation and the chosen solution

Render's default filesystem is ephemeral: no persistent disk exists unless
you attach one (paid plans, single service only — and it still would not be
shared between the API and worker containers).

**Decision: no new storage was introduced** (per the no-redesign constraint).
Rationale:

- Blog **content never depended on the filesystem** — `content`, `plan` and
  `evidence` live in Postgres and flow through `/jobs/{id}` and `/blogs`;
  the `outputs/` file is a redundant copy.
- The **only filesystem-dependent feature is generated PNG bytes**:
  acceptable to lose on restart because (a) images are opt-in and rare,
  (b) the article text, placeholders-turned-notes and captions survive in
  Postgres, (c) regeneration is one request away, and (d) jobs are cached
  per `(topic, as_of, images)`.
- Adding S3/R2 + signed URLs + DB schema changes + upload code would be a
  redesign of the image pipeline — explicitly out of scope. If durable
  images are ever required, that is the documented next step: upload PNGs
  from the worker node to object storage and persist the public URL in the
  Markdown instead of `/assets/images/…`.

Consequence to communicate to users: **old articles keep their text forever
(Postgres), but embedded `/assets/images/…` pictures may 404 after a
redeploy/restart; regenerate the topic to restore them.**

