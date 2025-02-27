from crewai import Agent
from ..core.config import Config
from ..data.metrics_tools import MetricsTools

class ResourceAgent:
    """Resource usage analysis agent"""
    
    @staticmethod
    def create():
        """Create the resource usage analysis agent"""
        return Agent(
            name="Resource Monitor",
            role="Resource Utilization Expert",
            goal="Monitor and analyze resource usage to optimize infrastructure and prevent bottlenecks",
            backstory="""You are an expert in infrastructure optimization and resource management.
            With your deep understanding of CPU, memory, disk, and network utilization, you can 
            identify inefficient resource usage, predict scaling needs, and recommend infrastructure 
            changes to maintain optimal performance under varying loads.""",
            tools=[
                MetricsTools.check_resource_usage
            ],
            llm_model=Config.OPENAI_MODEL,
            verbose=Config.AGENT_VERBOSE
        )