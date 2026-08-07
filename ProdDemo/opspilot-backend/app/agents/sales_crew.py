from crewai import Agent, Task, Crew, Process

from app.llm.router import get_llm
from app.agents.tools import web_search, crm_lookup, send_email


def build_sales_crew(lead: dict) -> Crew:
    researcher = Agent(
        role="Lead Researcher",
        goal="Enrich raw lead data with public company/contact info found via web search.",
        backstory="A meticulous OSINT-style researcher who verifies before reporting.",
        tools=[web_search, crm_lookup],
        llm=get_llm("fast"),          # cheap model — this is retrieval + light synthesis
        verbose=True,
    )

    qualifier = Agent(
        role="Deal Qualifier",
        goal="Score the lead 0-100 and decide whether it's worth an outreach attempt.",
        backstory="A pragmatic sales ops analyst who kills bad leads fast and protects rep time.",
        llm=get_llm("balanced"),
        verbose=True,
    )

    outreach_writer = Agent(
        role="Outreach Strategist",
        goal="Draft a short, specific, non-generic first-touch email for a qualified lead.",
        backstory="A senior AE known for cold emails that actually get replies.",
        tools=[send_email],
        llm=get_llm("powerful"),      # persuasive copy is the highest-stakes step here
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Research this lead and enrich it with public info: {lead}. "
            "Find company size, industry, and any recent news relevant to a sales pitch."
        ),
        expected_output="A structured summary of enriched lead data.",
        agent=researcher,
    )

    qualify_task = Task(
        description="Using the enriched research, score this lead 0-100 and justify the score in one sentence.",
        expected_output="A score and a one-sentence justification.",
        agent=qualifier,
        context=[research_task],
    )

    outreach_task = Task(
        description=(
            "If the lead scored above 50, draft a 3-4 sentence personalized outreach email. "
            "Reference one concrete fact from the research. If below 50, output 'SKIP - low score'."
        ),
        expected_output="Either a drafted email or a skip notice.",
        agent=outreach_writer,
        context=[research_task, qualify_task],
    )

    return Crew(
        agents=[researcher, qualifier, outreach_writer],
        tasks=[research_task, qualify_task, outreach_task],
        process=Process.sequential,
        verbose=True,
    )
