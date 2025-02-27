from crewai import Crew, Task
from datetime import datetime
import json

class CrewManager:
    """Manages the creation and execution of crews"""
    
    def __init__(self, agent_manager):
        """Initialize crew manager"""
        self.agent_manager = agent_manager
    
    def process_query(self, query):
        """
        Process a user query using agents
        
        Args:
            query: The user query
            
        Returns:
            The processed result
        """
        # Determine which agent should handle this query
        agent_type, main_agent = self.agent_manager.determine_agent_for_query(query)
        
        print(f"Selected agent type for query: {agent_type}")
        
        # Create task for the main agent
        task = Task(
            description=f"Analyze and provide insights for the following query: {query}",
            expected_output="A detailed analysis with insights and recommendations",
            agent=main_agent
        )
        
        # Create crew with just the main agent for simple queries
        crew = Crew(
            agents=[main_agent],
            tasks=[task],
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Format and return results
        return self._format_result(query, result, agent_type)
    
    def process_complex_query(self, query):
        """
        Process a complex query that might require multiple agents
        
        Args:
            query: The user query
            
        Returns:
            The processed result
        """
        # Get all agents
        all_agents = self.agent_manager.get_all_agents()
        
        # Create tasks for each agent type with specific focus
        performance_task = Task(
            description=f"Analyze performance aspects of: {query}",
            expected_output="Performance analysis and insights",
            agent=self.agent_manager.get_agent_by_type("performance")
        )
        
        error_task = Task(
            description=f"Analyze error patterns related to: {query}",
            expected_output="Error analysis and troubleshooting recommendations",
            agent=self.agent_manager.get_agent_by_type("error")
        )
        
        resource_task = Task(
            description=f"Analyze resource utilization for: {query}",
            expected_output="Resource usage analysis and optimization recommendations",
            agent=self.agent_manager.get_agent_by_type("resource")
        )
        
        business_task = Task(
            description=f"Analyze business impact of: {query}",
            expected_output="Business metrics analysis and strategic insights",
            agent=self.agent_manager.get_agent_by_type("business")
        )
        
        # Create a crew with all agents and their tasks
        crew = Crew(
            agents=all_agents,
            tasks=[performance_task, error_task, resource_task, business_task],
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Format and return results
        return self._format_result(query, result, "multi-agent")
    
    def _format_result(self, query, result, agent_type):
        """Format the result from crew execution"""
        return {
            "query": query,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }