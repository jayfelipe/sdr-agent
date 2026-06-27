import random
from langchain_core.tools import tool

DUMMY_NEWS = [
    "Recently raised a Series B funding round of $45M to expand operations.",
    "Just announced a new CEO transition last week.",
    "Opened a new headquarters in Austin, Texas.",
    "Suffered a minor data breach, highlighting cybersecurity needs.",
    "Launched a new AI-driven product feature last month."
]

@tool
def scrape_recent_news(company_name: str) -> str:
    """
    Use this tool to find the latest news or press releases about a lead's company.
    Helps to personalize the outreach email.
    """
    # TODO: Replace with a real scraping API like Firecrawl or SerpAPI when budget allows
    news = random.choice(DUMMY_NEWS)
    return f"News about {company_name}: {news}"