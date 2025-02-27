"""
Query Manager Agent Module

This module implements a CrewAI agent that orchestrates the entire analysis workflow
and synthesizes the final comprehensive response.
"""

import os
from typing import Dict, Any, Optional, List
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.metricsCrew.config import get_agent_config, get_prompt_template


class ManagerAgent:
    """
    Agent that orchestrates the entire analysis workflow.
    
    This agent serves as both the entry point and final integration point
    for the system. It manages the flow between specialized agents and
    synthesizes the final response.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ManagerAgent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        # Load configuration with defaults
        self.config = config or get_agent_config('manager_agent', {})
    
    def get_agent(self) -> Agent:
        """
        Create and configure the CrewAI agent.
        
        Returns:
            Configured CrewAI Agent instance
        """
        # Get LLM configuration
        llm_config = self.config.get('llm', {})
        model_name = llm_config.get('model_name', 'gemini-pro')
        temperature = llm_config.get('temperature', 0.2)
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Get prompt template for response synthesis
        synthesis_prompt = get_prompt_template('manager_agent.prompt', 
            """You are a Business Analytics Director who orchestrates data analysis
            and synthesizes insights into comprehensive responses. Your task is to:
            
            1. Understand business questions and direct them to appropriate specialists
            2. Ensure all analysis components work together cohesively
            3. Integrate technical results with business context
            4. Synthesize a clear, comprehensive response that directly addresses the original question
            5. Highlight key insights and business implications
            
            Your responses should be business-focused, actionable, and directly address
            the question being asked, translating technical analysis into business value.""")
        
        # Create the agent
        agent = Agent(
            role=self.config.get('role', "Business Analytics Director"),
            goal=self.config.get('goal', "Orchestrate analysis and synthesize comprehensive responses"),
            backstory=self.config.get('backstory', 
                "A strategic executive with expertise in translating business needs into actionable analytics"),
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', True),
            llm=llm
        )
        
        return agent
    
    def synthesize_response(self, query: str, schema_info: Dict[str, Any],
                          query_interpretation: Dict[str, Any],
                          execution_results: Dict[str, Any],
                          business_interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize a comprehensive response from all analysis components.
        
        This is a utility method that directly uses the LLM without going through
        the CrewAI task framework.
        
        Args:
            query: Original user query
            schema_info: Dataset schema information
            query_interpretation: Structured analysis requirements
            execution_results: Code execution results
            business_interpretation: Business interpretation of results
            
        Returns:
            Dictionary with synthesized response
        """
        try:
            # Get LLM configuration
            llm_config = self.config.get('llm', {})
            model_name = llm_config.get('model_name', 'gemini-pro')
            temperature = llm_config.get('temperature', 0.3)
            
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
            
            # Create the prompt for response synthesis
            import json
            
            prompt = f"""You are a Business Analytics Director synthesizing a comprehensive response
            to a business question based on data analysis.
            
            Original User Query:
            "{query}"
            
            Dataset Schema Information:
            {json.dumps(schema_info, indent=2)}
            
            Query Interpretation:
            {json.dumps(query_interpretation, indent=2)}
            
            Analysis Results:
            {json.dumps(execution_results.get('result', {}), indent=2)}
            
            Business Interpretation:
            {json.dumps(business_interpretation, indent=2)}
            
            Your task is to synthesize a comprehensive response that:
            1. Directly answers the original query
            2. Provides context and explanation for the findings
            3. Highlights key insights and their business implications
            4. Structures the information in a clear, logical flow
            5. Uses business language accessible to non-technical users
            
            Format your response as a comprehensive business answer with these sections:
            1. Executive Summary: Brief answer to the query
            2. Key Findings: Detailed explanation of results
            3. Business Implications: What these results mean for the business
            4. Recommendations: Suggested actions based on the analysis
            
            The response should be professional, insightful, and actionable.
            """
            
            # Get the response from the LLM
            response = llm.invoke(prompt)
            
            # Return the synthesized response
            return {
                "response": response.content,
                "query": query
            }
            
        except Exception as e:
            return {"error": f"Error synthesizing response: {str(e)}"}