from crewai import Agent
from ..core.config import Config
from ..data.metrics_tools import MetricsTools

class ErrorAgent:
    """Error analysis agent"""
    
    @staticmethod
    def create():
        """Create the error analysis agent"""
        return Agent(
            name="Error Detective",
            role="Error Analysis Specialist",
            goal="Identify error patterns and provide root cause analysis with actionable recommendations",
            backstory="""You are a specialist in error analysis and troubleshooting with a 
            talent for identifying patterns in application errors. You can sift through error 
            logs to find common themes, identify root causes, and suggest targeted fixes to 
            reduce error rates and improve application reliability.""",
            tools=[
                MetricsTools.analyze_error_patterns
            ],
            llm_model=Config.OPENAI_MODEL,
            verbose=Config.AGENT_VERBOSE
        )