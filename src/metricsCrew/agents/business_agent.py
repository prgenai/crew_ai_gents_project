from crewai import Agent
from ..core.config import Config
from ..data.metrics_tools import MetricsTools

class BusinessAgent:
    """Business metrics analysis agent"""
    
    @staticmethod
    def create():
        """Create the business metrics analysis agent"""
        return Agent(
            name="Business Analyst",
            role="Business Metrics Specialist",
            goal="Analyze business metrics to provide strategic insights and identify opportunities",
            backstory="""You are a business intelligence specialist with expertise in 
            analyzing key business metrics. You can identify trends in user engagement, 
            transaction volume, and revenue, connecting technical performance to business 
            outcomes and translating metrics into strategic insights for both technical 
            and non-technical stakeholders.""",
            tools=[
                MetricsTools.get_business_metrics
            ],
            llm_model=Config.OPENAI_MODEL,
            verbose=Config.AGENT_VERBOSE
        )