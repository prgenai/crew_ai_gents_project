from crewai import Agent
from ..core.config import Config
from ..data.metrics_tools import MetricsTools

class PerformanceAgent:
    """Performance metrics analysis agent"""
    
    @staticmethod
    def create():
        """Create the performance analysis agent"""
        return Agent(
            name="Performance Analyst",
            role="Application Performance Expert",
            goal="Analyze application performance metrics and provide actionable insights",
            backstory="""You are an expert in application performance analysis with years of 
            experience optimizing high-traffic systems. You can quickly identify performance 
            bottlenecks, understand response time patterns, and provide recommendations for 
            improving application speed and reliability.""",
            tools=[
                MetricsTools.fetch_performance_metrics
            ],
            llm_model=Config.OPENAI_MODEL,
            verbose=Config.AGENT_VERBOSE
        )