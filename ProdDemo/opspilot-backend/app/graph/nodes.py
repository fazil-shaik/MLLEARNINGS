"""
Each node is a plain function: (state) -> partial state update.
Classifier uses the 'fast' tier since it's just routing, not reasoning.
"""
import time

from app.graph.state import OpsPilotState
from app.llm.router import get_llm
from app.agents.sales_crew import build_sales_crew
from app.agents.support_crew import build_support_crew
from app.agents.ops_crew import build_ops_crew


def classify_event(state: OpsPilotState) -> dict:
    """event_type already comes from the API layer (which endpoint was hit),
    so this node mainly maps it to a department and can be extended to do
    smarter routing later (e.g. free-text intent classification)."""
    mapping = {"lead": "sales", "ticket": "support", "ops_task": "ops"}
    department = mapping.get(state["event_type"], "ops")
    return {"department": department}


def run_sales_crew(state: OpsPilotState) -> dict:
    crew = build_sales_crew(state["payload"])
    result = crew.kickoff()
    return {"crew_result": str(result)}


def run_support_crew(state: OpsPilotState) -> dict:
    crew = build_support_crew(state["payload"])
    result = crew.kickoff()
    needs_human = "ESCALATE" in str(result).upper()
    return {"crew_result": str(result), "needs_human": needs_human}


def run_ops_crew(state: OpsPilotState) -> dict:
    crew = build_ops_crew(state["payload"])
    result = crew.kickoff()
    return {"crew_result": str(result)}


def route_by_department(state: OpsPilotState) -> str:
    return state["department"]


async def persist_result(state: OpsPilotState) -> dict:
    """
    Writes two things to NeonDB:
    1. The domain record (leads/tickets/ops_tasks) so it's queryable by the API —
       tickets that need a human are stored with status='escalated'.
    2. An agent_runs row for the audit trail.
    Returns db_record_id so the API layer can hand the caller a real id.
    """
    from app.database.connection import AsyncSessionLocal
    from app.database.models import AgentRun, Lead, Ticket, OpsTask

    department = state.get("department")
    payload = state.get("payload", {})
    crew_result = str(state.get("crew_result"))
    needs_human = state.get("needs_human", False)
    db_record_id = None

    async with AsyncSessionLocal() as session:
        if department == "sales":
            record = Lead(
                name=payload.get("name", "Unknown"),
                company=payload.get("company"),
                email=payload.get("email"),
                source=payload.get("source"),
                status="contacted",
                outreach_draft=crew_result,
            )
            session.add(record)
            await session.flush()
            db_record_id = record.id

        elif department == "support":
            record = Ticket(
                customer=payload.get("customer", "Unknown"),
                subject=payload.get("subject"),
                description=payload.get("description"),
                status="escalated" if needs_human else "resolved",
                resolution={"draft": crew_result},
            )
            session.add(record)
            await session.flush()
            db_record_id = record.id

        elif department == "ops":
            record = OpsTask(
                title=payload.get("title", "Untitled"),
                description=payload.get("description"),
                status="done",
                result={"report": crew_result},
            )
            session.add(record)
            await session.flush()
            db_record_id = record.id

        run = AgentRun(
            thread_id=state.get("thread_id"),
            crew_name=department,
            agent_name="crew_pipeline",
            model_used="mixed-tier",
            input_summary=str(payload)[:500],
            output_summary=crew_result[:2000],
        )
        session.add(run)
        await session.commit()

    return {"db_record_id": db_record_id}
