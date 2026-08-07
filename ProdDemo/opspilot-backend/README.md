# OpsPilot — Multi-Agent Business Automation Backend

Event-driven backend where a **LangGraph** supervisor routes business events
(new lead / support ticket / ops request) to specialized **CrewAI** crews.
Agents call **OpenRouter** models (tiered by cost/capability) and use
**LangChain** tools (web search etc.) for real-time info. **NeonDB**
(serverless Postgres) stores business data, the audit trail of every agent
run, and the LangGraph checkpoints themselves — so every run is durable and
resumable by `thread_id`.

## Architecture

```
Event (API) ─▶ classify_event ─▶ route ─┬─▶ sales_node   (CrewAI: researcher → qualifier → outreach writer)
                                          ├─▶ support_node (CrewAI: triage → resolver → escalation judge)
                                          └─▶ ops_node     (CrewAI: planner → executor → reporter)
                                                    │
                                                    ▼
                                            persist_result ─▶ NeonDB (agent_runs)
```

- `app/graph/orchestrator.py` — the LangGraph `StateGraph`, checkpointed to NeonDB.
- `app/agents/*_crew.py` — the three CrewAI crews (one per department).
- `app/llm/router.py` — OpenRouter model tiers (`fast` / `balanced` / `powerful`) shared by every agent.
- `app/agents/tools.py` — shared LangChain tools (web search via Tavily; CRM/email/KB are stubs to wire up).
- `app/database/` — SQLAlchemy models + raw schema for NeonDB.
- `app/api/routes.py` — FastAPI endpoints that fire graph runs.

## Why this shape

- **Cost control**: triage/classification agents run on cheap fast models; only the
  highest-stakes step per crew (persuasive copy, escalation judgment) uses the powerful tier.
  Change tiers in `.env` with zero code changes.
- **Durability & audit**: LangGraph's Postgres checkpointer + the `agent_runs` table mean
  every run can be replayed/inspected by `thread_id`, not just fire-and-forget.
- **Swap-friendly tools**: `crm_lookup`, `send_email`, `kb_search` are stubs on purpose —
  wire them to your real CRM/email/vector-store without touching the graph or crews.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OpenRouter key, NeonDB URLs, Tavily key
uvicorn app.main:app --reload
```

NeonDB: create a project at neon.tech, grab the **pooled** connection string for
`DATABASE_URL` and the **direct/unpooled** one for `DATABASE_URL_UNPOOLED`
(LangGraph's checkpointer needs a direct connection). Tables are created
automatically on startup (`init_schema()` + LangGraph's own checkpoint tables).

OpenRouter: create a key at openrouter.ai, set `OPENROUTER_API_KEY`. Model slugs
in `.env` can be any OpenRouter-supported model — mix providers freely.

## Try it

```bash
curl -X POST localhost:8000/events/lead \
  -H "Content-Type: application/json" \
  -d '{"name": "Jane Doe", "company": "Acme Robotics", "email": "jane@acme.io", "source": "website"}'

curl -X POST localhost:8000/events/ticket \
  -H "Content-Type: application/json" \
  -d '{"customer": "Bob", "subject": "Can'\''t log in", "description": "2FA code never arrives"}'

curl -X POST localhost:8000/events/ops-task \
  -H "Content-Type: application/json" \
  -d '{"title": "Set up weekly competitor pricing report"}'
```

## Human review workflow

When the Support crew's Escalation Judge decides `ESCALATE`, the ticket is
persisted with `status="escalated"` instead of being auto-resolved.

- `GET /tickets/pending-review` — queue of tickets waiting on a human.
- `GET /tickets/{id}` — full ticket detail, including the agent's draft resolution.
- `POST /tickets/{id}/review` — human decides:
  - `{"action": "approve"}` — send the agent's draft as-is → `status=resolved`
  - `{"action": "edit", "edited_response": "..."}` — human's text is the final response → `status=resolved`
  - `{"action": "reject"}` — discard the draft, ticket goes back to `status=open` for manual handling

```bash
curl localhost:8000/tickets/pending-review

curl -X POST localhost:8000/tickets/3/review \
  -H "Content-Type: application/json" \
  -d '{"action": "edit", "edited_response": "Hi Bob, resetting your 2FA now...", "reviewer_note": "draft missed the account-lock edge case"}'
```

## Next steps (not yet built)

- Wire `crm_lookup` / `send_email` / `kb_search` to real providers.
- Actually *send* the approved/edited response (currently the review endpoint just updates DB state — wire it to `send_email`).
- Add auth + rate limiting to the FastAPI layer before exposing publicly.
- Frontend (coming next): a dashboard reading `leads` / `tickets` / `ops_tasks` /
  `agent_runs` from NeonDB, plus a way to trigger events and watch a run live.
