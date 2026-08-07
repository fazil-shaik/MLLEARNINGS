from crewai import Agent, Task, Crew, Process

from app.llm.router import get_llm
from app.agents.tools import web_search, kb_search


def build_support_crew(ticket: dict) -> Crew:
    triage = Agent(
        role="Ticket Triage Specialist",
        goal="Classify ticket category and priority quickly and consistently.",
        backstory="A support lead who has seen every ticket type and triages in seconds.",
        llm=get_llm("fast"),
        verbose=True,
    )

    resolver = Agent(
        role="Resolution Agent",
        goal="Find or draft a solution for the customer's issue using KB and web search.",
        backstory="A senior support engineer who writes clear, empathetic, correct answers.",
        tools=[kb_search, web_search],
        llm=get_llm("balanced"),
        verbose=True,
    )

    escalation = Agent(
        role="Escalation Judge",
        goal="Decide if this needs a human, and why.",
        backstory="A support manager who protects customers from bad automated answers.",
        llm=get_llm("powerful"),
        verbose=True,
    )

    triage_task = Task(
        description=f"Classify this ticket's category and priority (low/normal/high/urgent): {ticket}",
        expected_output="category: <value>, priority: <value>",
        agent=triage,
    )

    resolve_task = Task(
        description="Attempt to resolve the ticket using KB search and, if needed, web search for public docs.",
        expected_output="A draft customer-facing response, or 'NO_RESOLUTION_FOUND'.",
        agent=resolver,
        context=[triage_task],
    )

    escalate_task = Task(
        description=(
            "Given the triage and draft resolution, decide: RESOLVE (send the draft as-is), "
            "REVISE (draft needs human edit first), or ESCALATE (route to a human agent now). "
            "Give a one-sentence reason."
        ),
        expected_output="decision: <RESOLVE|REVISE|ESCALATE>, reason: <one sentence>",
        agent=escalation,
        context=[triage_task, resolve_task],
    )

    return Crew(
        agents=[triage, resolver, escalation],
        tasks=[triage_task, resolve_task, escalate_task],
        process=Process.sequential,
        verbose=True,
    )
