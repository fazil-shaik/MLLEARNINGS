from crewai import Agent, Task, Crew, Process

from app.llm.router import get_llm
from app.agents.tools import web_search


def build_ops_crew(request: dict) -> Crew:
    planner = Agent(
        role="Ops Task Planner",
        goal="Break an operations request down into a concrete, ordered checklist of steps.",
        backstory="A COO's right hand who turns vague asks into executable plans.",
        llm=get_llm("balanced"),
        verbose=True,
    )

    executor = Agent(
        role="Automation Executor",
        goal="Work through the plan step by step, using web search for anything requiring current info.",
        backstory="A detail-oriented operator who actually finishes the checklist instead of just describing it.",
        tools=[web_search],
        llm=get_llm("fast"),
        verbose=True,
    )

    reporter = Agent(
        role="Ops Reporter",
        goal="Summarize what was done, what's still open, and any risks, for a non-technical stakeholder.",
        backstory="Writes status updates people actually read.",
        llm=get_llm("balanced"),
        verbose=True,
    )

    plan_task = Task(
        description=f"Turn this ops request into a numbered step-by-step plan: {request}",
        expected_output="A numbered list of concrete steps.",
        agent=planner,
    )

    execute_task = Task(
        description="Execute the plan's steps as far as possible with available tools. Note any step that needs a human.",
        expected_output="Results per step, and a list of steps still requiring a human.",
        agent=executor,
        context=[plan_task],
    )

    report_task = Task(
        description="Write a short status report: done, pending, risks/blockers.",
        expected_output="A 3-part status report (Done / Pending / Risks).",
        agent=reporter,
        context=[plan_task, execute_task],
    )

    return Crew(
        agents=[planner, executor, reporter],
        tasks=[plan_task, execute_task, report_task],
        process=Process.sequential,
        verbose=True,
    )
