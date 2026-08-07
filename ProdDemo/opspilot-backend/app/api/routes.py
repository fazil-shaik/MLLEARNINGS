import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.graph.orchestrator import run_event
from app.database.connection import AsyncSessionLocal
from app.database.models import Ticket

router = APIRouter()


class LeadIn(BaseModel):
    name: str
    company: str | None = None
    email: str | None = None
    source: str | None = None


class TicketIn(BaseModel):
    customer: str
    subject: str | None = None
    description: str | None = None


class OpsTaskIn(BaseModel):
    title: str
    description: str | None = None


@router.post("/events/lead")
async def create_lead(lead: LeadIn):
    thread_id = f"lead-{uuid.uuid4()}"
    result = await run_event(thread_id, "lead", lead.model_dump())
    return {"thread_id": thread_id, "result": result["crew_result"]}


@router.post("/events/ticket")
async def create_ticket(ticket: TicketIn):
    thread_id = f"ticket-{uuid.uuid4()}"
    result = await run_event(thread_id, "ticket", ticket.model_dump())
    return {
        "thread_id": thread_id,
        "ticket_id": result.get("db_record_id"),
        "result": result["crew_result"],
        "needs_human": result.get("needs_human", False),
    }


# ---------- Human review queue ----------

class TicketOut(BaseModel):
    id: int
    customer: str
    subject: str | None
    description: str | None
    status: str
    resolution: dict

    class Config:
        from_attributes = True


class ReviewIn(BaseModel):
    action: Literal["approve", "edit", "reject"]
    # required when action == "edit": the human-corrected response to send instead
    edited_response: str | None = None
    reviewer_note: str | None = None


@router.get("/tickets/pending-review", response_model=list[TicketOut])
async def list_pending_review():
    """Tickets the escalation agent flagged as needing a human before anything is sent."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Ticket).where(Ticket.status == "escalated").order_by(Ticket.created_at)
        )
        return result.scalars().all()


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(ticket_id: int):
    async with AsyncSessionLocal() as session:
        ticket = await session.get(Ticket, ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket


@router.post("/tickets/{ticket_id}/review", response_model=TicketOut)
async def review_ticket(ticket_id: int, review: ReviewIn):
    """
    Human resolves an escalated ticket.
    - approve: send the agent's draft as-is -> status=resolved
    - edit:    human supplies edited_response, that's what gets sent -> status=resolved
    - reject:  agent's draft is discarded, ticket stays open for a human to handle manually
    """
    if review.action == "edit" and not review.edited_response:
        raise HTTPException(status_code=400, detail="edited_response is required when action='edit'")

    async with AsyncSessionLocal() as session:
        ticket = await session.get(Ticket, ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if ticket.status != "escalated":
            raise HTTPException(status_code=400, detail=f"Ticket is not pending review (status={ticket.status})")

        resolution = dict(ticket.resolution or {})
        resolution["reviewer_note"] = review.reviewer_note
        resolution["review_action"] = review.action

        if review.action == "approve":
            ticket.status = "resolved"
        elif review.action == "edit":
            resolution["final_response"] = review.edited_response
            ticket.status = "resolved"
        else:  # reject
            ticket.status = "open"

        ticket.resolution = resolution
        await session.commit()
        await session.refresh(ticket)
        return ticket


@router.post("/events/ops-task")
async def create_ops_task(task: OpsTaskIn):
    thread_id = f"ops-{uuid.uuid4()}"
    result = await run_event(thread_id, "ops_task", task.model_dump())
    return {"thread_id": thread_id, "result": result["crew_result"]}
