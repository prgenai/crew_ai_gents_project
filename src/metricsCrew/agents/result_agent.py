"""
Result Interpretation Agent Module

This module implements a CrewAI agent specialized in executing analytical code
and interpreting the results in business context.
"""

import os
import json
from typing import Dict, Any, Optional, List
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.metricsCrew.core.data_manager import DataManager
from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.tools.result_tool import ResultAnalysisTool
from src.metricsCrew.config import get_agent_config, get_prompt_template


class ResultAgent:
    """
    Agent specialized in interpreting analysis results in business terms.
    
    This agent executes the generated code against the dataset and interprets
    the results in a business context, providing insights and explanations
    that are meaningful to business users.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                data_manager: Optional[DataManager] = None,
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the ResultAgent.
        
        Args:
            config: Configuration dictionary for the agent
            data_manager: DataManager instance for dataset access
            schema_registry: SchemaRegistry instance for metadata access
        """
        # Load configuration with defaults
        self.config = config or get_agent_config('result_agent', {})
        
        # Set up dependencies
        self.data_manager = data_manager or DataManager()
        self.schema_registry = schema_registry or SchemaRegistry()
        
        # Create the result analysis tool
        self.result_tool = ResultAnalysisTool(
            data_manager=self.data_manager,
            schema_registry=self.schema_registry
        )
    
    def get_agent(self) -> Agent:
        """
        Create and configure the CrewAI agent.
        
        Returns:
            Configured CrewAI Agent instance
        """
        # Get LLM configuration
        llm_config = self.config.get('llm', {})
        model_name = llm_config.get('model_name', 'gemini-pro')
        temperature = llm_config.get('temperature', 0.4)
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Get prompt template for result interpretation
        result_prompt = get_prompt_template('result_agent.prompt', 
            """You are a Business Intelligence Analyst specializing in interpreting data analysis results.
            Your task is to explain technical results in business-friendly terms, highlighting key insights
            and implications. For each analysis:
            
            1. Identify the most important findings
            2. Explain what the results mean in business terms
            3. Highlight any anomalies or unexpected patterns
            4. Suggest potential business implications
            5. Use clear, non-technical language accessible to business users
            
            Your explanations should be thorough yet concise, focusing on the "so what" 
            of the analysis rather than technical details.""")
        
        # Create the agent
        agent = Agent(
            role=self.config.get('role', "Business Intelligence Analyst"),
            goal=self.config.get('goal', "Explain technical results in business-friendly terms"),
            backstory=self.config.get('backstory', 
                "A communication specialist who bridges technical analysis and business decision-making"),
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', False),
            llm=llm,
            tools=[self.result_tool]
        )
        
        return agent
    
    def analyze_results(self, code: str, requirements: Dict[str, Any], 
                       dataset_id: str, query: str) -> Dict[str, Any]:
        """
        Execute code and interpret the results in business context.
        
        This is a utility method that directly uses the result tool without
        going through the CrewAI task framework.
        
        Args:
            code: Python code to execute
            requirements: Analytical requirements dictionary
            dataset_id: Identifier for the dataset
            query: Original user query
            
        Returns:
            Dictionary with analysis results and business interpretation
        """
        # Use the result tool to execute the code and get raw results
        execution_result = self.result_tool.execute_code(
            code=code,
            dataset_id=dataset_id
        )
        
        # Check for execution errors
        if "error" in execution_result:
            return execution_result
        
        # Interpret the results
        interpretation = self.result_tool.interpret_results(
            results=execution_result,
            requirements=requirements,
            query=query,
            dataset_id=dataset_id
        )
        
        # Combine execution results and interpretation
        return {
            "execution_results": execution_result,
            "business_interpretation": interpretation,
            "requirements": requirements,
            "query": query
        }