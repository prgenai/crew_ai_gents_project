"""
Task Definitions Module

This module defines the CrewAI tasks that form the business data analysis workflow.
"""

from typing import Dict, Any, Optional, List
from crewai import Task

from src.metricsCrew.config import get_task_config


def create_schema_analysis_task(agent, dataset_path: str, dataset_id: Optional[str] = None,
                             description: Optional[str] = None) -> Task:
    """
    Create a task for schema analysis.
    
    Args:
        agent: Schema Analysis Agent instance
        dataset_path: Path to the dataset file
        dataset_id: Optional identifier for the dataset
        description: Optional description of the dataset
        
    Returns:
        Configured CrewAI Task
    """
    config = get_task_config('schema_analysis', {})
    
    # Format the task description with dataset information
    description = description or "Dataset for analysis"
    task_description = f"""
    Analyze the structure of the dataset located at '{dataset_path}'.
    Identify columns, data types, and relationships in the dataset.
    Classify columns as metrics, dimensions, or timestamps.
    Create a business glossary mapping technical fields to business terms.
    
    Dataset ID: {dataset_id or 'Will be determined automatically'}
    Dataset Description: {description}
    """
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=config.get('expected_output', 
            "Detailed schema information including column metadata, relationships, and business glossary"),
        agent=agent
    )
    
    return task


def create_query_interpretation_task(agent, query: str, schema_info: Dict[str, Any]) -> Task:
    """
    Create a task for query interpretation.
    
    Args:
        agent: Query Interpretation Agent instance
        query: Natural language business query
        schema_info: Dataset schema information
        
    Returns:
        Configured CrewAI Task
    """
    config = get_task_config('query_interpretation', {})
    
    # Format the task description with query information
    task_description = f"""
    Interpret the following business query and translate it into specific analytical requirements:
    
    User Query: "{query}"
    
    Use the provided schema information to map business terms to technical fields.
    Identify requested metrics, dimensions, filters, and analysis type.
    Structure the requirements for code generation.
    """
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=config.get('expected_output', 
            "Structured JSON with analysis requirements including metrics, dimensions, filters, and analysis type"),
        agent=agent,
        context=[
            {"role": "schema_info", "content": schema_info}
        ]
    )
    
    return task


def create_code_generation_task(agent, requirements: Dict[str, Any], schema_info: Dict[str, Any]) -> Task:
    """
    Create a task for code generation.
    
    Args:
        agent: Code Generation Agent instance
        requirements: Structured analysis requirements
        schema_info: Dataset schema information
        
    Returns:
        Configured CrewAI Task
    """
    config = get_task_config('code_generation', {})
    
    # Format the task description with requirements information
    import json
    task_description = f"""
    Create pandas code to perform the following analysis:
    
    Analysis Requirements:
    {json.dumps(requirements, indent=2)}
    
    Use the provided schema information to understand the dataset structure.
    Generate clean, efficient, and well-commented pandas code.
    Ensure code handles edge cases and follows best practices.
    Return the code as a Python function named 'analyze_data'.
    """
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=config.get('expected_output', 
            "Executable Python code that performs the requested analysis"),
        agent=agent,
        context=[
            {"role": "requirements", "content": requirements},
            {"role": "schema_info", "content": schema_info}
        ]
    )
    
    return task


def create_result_analysis_task(agent, code: str, requirements: Dict[str, Any], 
                             dataset_id: str, query: str) -> Task:
    """
    Create a task for result analysis.
    
    Args:
        agent: Result Interpretation Agent instance
        code: Generated Python code
        requirements: Structured analysis requirements
        dataset_id: Identifier for the dataset
        query: Original user query
        
    Returns:
        Configured CrewAI Task
    """
    config = get_task_config('result_analysis', {})
    
    # Format the task description with code and query information
    task_description = f"""
    Execute the following code against dataset '{dataset_id}' and interpret the results:
    
    User Query: "{query}"
    
    The code performs analysis based on the provided requirements.
    Execute the code and capture the results.
    Interpret the results in business context.
    Highlight key insights and implications.
    Explain what the data shows in business-friendly terms.
    """
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=config.get('expected_output', 
            "Business interpretation and visualizations of analysis results"),
        agent=agent,
        context=[
            {"role": "code", "content": code},
            {"role": "requirements", "content": requirements},
            {"role": "dataset_id", "content": dataset_id},
            {"role": "query", "content": query}
        ]
    )
    
    return task


def create_response_synthesis_task(agent, query: str, schema_info: Dict[str, Any],
                                query_interpretation: Dict[str, Any],
                                execution_results: Dict[str, Any],
                                business_interpretation: Dict[str, Any]) -> Task:
    """
    Create a task for response synthesis.
    
    Args:
        agent: Query Manager Agent instance
        query: Original user query
        schema_info: Dataset schema information
        query_interpretation: Structured analysis requirements
        execution_results: Code execution results
        business_interpretation: Business interpretation of results
        
    Returns:
        Configured CrewAI Task
    """
    config = get_task_config('response_synthesis', {})
    
    # Format the task description
    task_description = f"""
    Synthesize a comprehensive response to the following query:
    
    User Query: "{query}"
    
    Integrate all analysis components into a cohesive response.
    Ensure the response directly addresses the original query.
    Highlight key insights and business implications.
    Structure the response for clarity and impact.
    Use business language accessible to non-technical users.
    """
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=config.get('expected_output', 
            "Comprehensive answer to user's query in business language"),
        agent=agent,
        context=[
            {"role": "query", "content": query},
            {"role": "schema_info", "content": schema_info},
            {"role": "query_interpretation", "content": query_interpretation},
            {"role": "execution_results", "content": execution_results},
            {"role": "business_interpretation", "content": business_interpretation}
        ]
    )
    
    return task