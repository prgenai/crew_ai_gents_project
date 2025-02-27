from .performance_agent import PerformanceAgent
from .error_agent import ErrorAgent
from .resource_agent import ResourceAgent
from .business_agent import BusinessAgent
from ..core.config import Config
import openai

class AgentManager:
    """Manages all agents and their interactions"""
    
    def __init__(self):
        """Initialize agent manager"""
        # Initialize OpenAI
        openai.api_key = Config.OPENAI_API_KEY
        
        # Create agents
        self.performance_agent = PerformanceAgent.create()
        self.error_agent = ErrorAgent.create()
        self.resource_agent = ResourceAgent.create()
        self.business_agent = BusinessAgent.create()
        
        # Store all agents in a dict for easy access
        self.agents = {
            "performance": self.performance_agent,
            "error": self.error_agent,
            "resource": self.resource_agent,
            "business": self.business_agent
        }
    
    def get_agent_by_type(self, agent_type):
        """Get agent by type"""
        return self.agents.get(agent_type)
    
    def get_all_agents(self):
        """Get all agents"""
        return list(self.agents.values())
    
    def determine_agent_for_query(self, query):
        """
        Determine which agent should handle a query
        
        Args:
            query: The user query
            
        Returns:
            A tuple of (agent_type, agent)
        """
        # Use OpenAI to classify the query
        prompt = f"""
        Analyze this query about application metrics and classify it into exactly ONE of these categories:
        - performance: for queries about response times, latency, throughput, etc.
        - error: for queries about error rates, exceptions, issues, incidents, etc.
        - resource: for queries about CPU, memory, disk usage, infrastructure, etc.
        - business: for queries about business metrics, users, revenue, transactions, etc.
        
        Query: {query}
        
        Respond with ONLY the category name (performance, error, resource, or business).
        """
        
        try:
            response = openai.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a query classifier for application metrics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Handle potential unexpected responses
            if classification not in self.agents:
                # Default to performance if classification fails
                print(f"Unexpected classification: {classification}. Defaulting to performance.")
                classification = "performance"
            
            return (classification, self.agents[classification])
        
        except Exception as e:
            print(f"Error classifying query: {e}")
            # Default to performance if error occurs
            return ("performance", self.performance_agent)