import random
from langchain_core.tools import tool

# Dummy data to simulate a real CRM response
DUMMY_CRM_DATA = [
    {"company": "GlobalTech", "industry": "Logistics", "employee_count": "500-1000", "tech_stack": "Salesforce, SAP"},
    {"company": "Innovatech", "industry": "E-commerce", "employee_count": "50-200", "tech_stack": "HubSpot, Shopify"},
    {"company": "DataDrive", "industry": "FinTech", "employee_count": "200-500", "tech_stack": "Zendesk, Stripe"},
]

@tool
def fetch_lead_crm_data(company_name: str) -> str:
    """
    Use this tool to fetch firmographic data about a lead's company from the CRM.
    Returns industry, employee count, and current tech stack.
    """
    
    # find a match or return a random one
    match = next((item for item in DUMMY_CRM_DATA if item["company"].lower() in company_name.lower()), None)
    
    if not match:
        match = random.choice(DUMMY_CRM_DATA) # Fallback to random data
        
    return f"Company: {match['company']}, Industry: {match['industry']}, Size: {match['employee_count']}, Tech Stack: {match['tech_stack']}"