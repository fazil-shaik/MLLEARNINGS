"""
Tools shared across crews. CrewAI agents accept LangChain tools directly.

- web_search: real-time info via Tavily (swap for SerpAPI/Bing if you prefer).
- crm_lookup / send_email: stubs — wire these to your real CRM/email provider.
  Keeping them as separate functions means each crew only needs to import
  what it actually uses.
"""
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

from app.config import settings

web_search = TavilySearchResults(
    max_results=5,
    tavily_api_key=settings.tavily_api_key,
)


@tool("crm_lookup")
def crm_lookup(company_name: str) -> str:
    """Look up existing CRM records for a company by name. STUB — wire to your real CRM."""
    return f"No existing CRM record found for '{company_name}'. Treat as a fresh lead."


@tool("send_email")
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. STUB — wire to SES/Sendgrid/Gmail API. Currently just logs."""
    return f"[stub] Would send email to={to} subject={subject!r} ({len(body)} chars)"


@tool("kb_search")
def kb_search(query: str) -> str:
    """Search internal knowledge base for support articles. STUB — wire to your real KB/vector store."""
    return f"[stub] No KB integration configured yet. Query was: {query}"
