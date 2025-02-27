"""
Query Interpretation Agent Module

This module implements a CrewAI agent specialized in interpreting natural language
business queries and translating them into structured analytical requirements.
"""

import os
import json
from typing import Dict, Any, Optional, List
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.config import get_agent_config, get_prompt_template


class QueryAgent:
    """
    Agent specialized in interpreting business questions and queries.
    
    This agent analyzes natural language queries to identify metrics, dimensions,
    filters, and analysis types, mapping business terminology to technical fields
    using the SchemaRegistry.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the QueryAgent.
        
        Args:
            config: Configuration dictionary for the agent
            schema_registry: SchemaRegistry instance for metadata access
        """
        # Load configuration with defaults
        self.config = config or get_agent_config('query_agent', {})
        
        # Set up dependencies
        self.schema_registry = schema_registry or SchemaRegistry()
    
    def get_agent(self) -> Agent:
        """
        Create and configure the CrewAI agent.
        
        Returns:
            Configured CrewAI Agent instance
        """
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
        
        # Get prompt template for query interpretation
        query_prompt = get_prompt_template('query_agent.prompt', 
            """You are a Business Domain Expert specializing in data analysis.
            Your task is to interpret business questions and translate them into
            structured analytical requirements. For each query:
            
            1. Identify the metrics being requested (e.g., sales, profit, quantity)
            2. Identify dimensions for analysis (e.g., region, category, time period)
            3. Identify any filters or conditions
            4. Determine the appropriate analysis type (e.g., trend, comparison, aggregation)
            5. Map business terms to technical field names
            
            Provide your response as a structured JSON with the format described in your instructions.""")
        
        # Create the agent
        agent = Agent(
            role=self.config.get('role', "Business Domain Expert"),
            goal=self.config.get('goal', "Translate business questions into specific analytical requirements"),
            backstory=self.config.get('backstory', 
                "A business analyst with deep domain knowledge who excels at understanding explicit and implicit requirements"),
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', False),
            llm=llm
        )
        
        return agent
    
    def interpret_query(self, query: str, dataset_id: str) -> Dict[str, Any]:
        """
        Interpret a natural language query and translate it to analytical requirements.
        
        This is a utility method that directly uses the LLM without going through
        the CrewAI task framework.
        
        Args:
            query: Natural language business query
            dataset_id: Identifier for the dataset to query
            
        Returns:
            Dictionary with structured analytical requirements
        """
        try:
            # Get schema information
            schema = self.schema_registry.get_schema(dataset_id)
            
            # Get metrics, dimensions, and timestamps
            metrics = self.schema_registry.get_metrics(dataset_id)
            dimensions = self.schema_registry.get_dimensions(dataset_id)
            timestamps = self.schema_registry.get_timestamps(dataset_id)
            
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
            
            # Create the prompt for query interpretation
            prompt = f"""You are a Business Domain Expert specializing in data analysis.
            Your task is to interpret the following business query and translate it 
            into specific analytical requirements.
            
            User Query: "{query}"
            
            Dataset Schema Information:
            - Metrics: {', '.join(metrics)}
            - Dimensions: {', '.join(dimensions)}
            - Time Dimensions: {', '.join(timestamps)}
            
            Business Glossary:
            {json.dumps(self.schema_registry._business_glossary, indent=2)}
            
            Your job is to:
            1. Identify the metrics being requested (e.g., sales, profit, quantity)
            2. Identify dimensions for analysis (e.g., region, category, time period)
            3. Identify any filters or conditions
            4. Determine the appropriate analysis type (e.g., trend, comparison, aggregation)
            5. Map business terms to technical field names
            
            Provide your response as a structured JSON with the following format:
            {{
              "metrics": ["field1", "field2"],
              "dimensions": ["field3", "field4"],
              "filters": [{{"field": "field5", "operator": "=", "value": "value1"}}],
              "analysis_type": "trend",
              "time_period": {{"field": "date_field", "start": "2021-01-01", "end": "2021-12-31"}},
              "sort_by": {{"field": "field1", "direction": "desc"}},
              "limit": 10
            }}
            
            Only include fields in your response that are mentioned or implied in the query.
            Make sure all field names match the actual column names in the dataset.
            """
            
            # Get the response from the LLM
            response = llm.invoke(prompt)
            
            # Extract the JSON response
            try:
                import re
                
                # Extract JSON from the response
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                
                if json_match:
                    requirements = json.loads(json_match.group(1))
                    return requirements
                else:
                    return {"error": "Could not extract JSON from LLM response"}
                    
            except Exception as e:
                return {"error": f"Error parsing LLM response: {str(e)}",
                        "raw_response": response.content}
            
        except Exception as e:
            return {"error": f"Error interpreting query: {str(e)}"}
    
    def validate_requirements(self, requirements: Dict[str, Any], 
                             dataset_id: str) -> Dict[str, Any]:
        """
        Validate analytical requirements against the dataset schema.
        
        Args:
            requirements: Dictionary with analytical requirements
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary with validation results and corrected requirements
        """
        try:
            schema = self.schema_registry.get_schema(dataset_id)
            validation_results = {"valid": True, "errors": [], "warnings": [], 
                                  "corrected_requirements": requirements.copy()}
            
            # Check metrics
            if "metrics" in requirements:
                for metric in requirements["metrics"]:
                    if metric not in schema["columns"]:
                        validation_results["valid"] = False
                        validation_results["errors"].append(f"Metric '{metric}' not found in dataset")
                        
                        # Try to find a similar metric
                        metrics = self.schema_registry.get_metrics(dataset_id)
                        matches = self._find_similar_columns(metric, metrics)
                        
                        if matches:
                            validation_results["warnings"].append(
                                f"Suggested replacement for '{metric}': {matches[0]}"
                            )
                            # Replace with the best match
                            idx = validation_results["corrected_requirements"]["metrics"].index(metric)
                            validation_results["corrected_requirements"]["metrics"][idx] = matches[0]
            
            # Check dimensions
            if "dimensions" in requirements:
                for dimension in requirements["dimensions"]:
                    if dimension not in schema["columns"]:
                        validation_results["valid"] = False
                        validation_results["errors"].append(f"Dimension '{dimension}' not found in dataset")
                        
                        # Try to find a similar dimension
                        dimensions = self.schema_registry.get_dimensions(dataset_id)
                        matches = self._find_similar_columns(dimension, dimensions)
                        
                        if matches:
                            validation_results["warnings"].append(
                                f"Suggested replacement for '{dimension}': {matches[0]}"
                            )
                            # Replace with the best match
                            idx = validation_results["corrected_requirements"]["dimensions"].index(dimension)
                            validation_results["corrected_requirements"]["dimensions"][idx] = matches[0]
            
            # Check filters
            if "filters" in requirements:
                for i, filter_condition in enumerate(requirements["filters"]):
                    field = filter_condition.get("field")
                    if field and field not in schema["columns"]:
                        validation_results["valid"] = False
                        validation_results["errors"].append(f"Filter field '{field}' not found in dataset")
                        
                        # Try to find a similar field
                        all_columns = list(schema["columns"].keys())
                        matches = self._find_similar_columns(field, all_columns)
                        
                        if matches:
                            validation_results["warnings"].append(
                                f"Suggested replacement for '{field}': {matches[0]}"
                            )
                            # Replace with the best match
                            validation_results["corrected_requirements"]["filters"][i]["field"] = matches[0]
            
            # Check time period
            if "time_period" in requirements and "field" in requirements["time_period"]:
                time_field = requirements["time_period"]["field"]
                if time_field not in schema["columns"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Time field '{time_field}' not found in dataset")
                    
                    # Try to find a similar time field
                    timestamps = self.schema_registry.get_timestamps(dataset_id)
                    matches = self._find_similar_columns(time_field, timestamps)
                    
                    if matches:
                        validation_results["warnings"].append(
                            f"Suggested replacement for '{time_field}': {matches[0]}"
                        )
                        # Replace with the best match
                        validation_results["corrected_requirements"]["time_period"]["field"] = matches[0]
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"],
                    "corrected_requirements": requirements}
    
    def _find_similar_columns(self, column: str, available_columns: List[str]) -> List[str]:
        """
        Find columns with similar names to the given column.
        
        Args:
            column: Column name to find matches for
            available_columns: List of available column names
            
        Returns:
            List of similar column names, sorted by similarity
        """
        # Simple implementation using string distance
        if not available_columns:
            return []
            
        # Calculate Levenshtein distance for each column
        try:
            from Levenshtein import distance
            distances = [(col, distance(column.lower(), col.lower())) for col in available_columns]
            
            # Sort by distance (smaller is better)
            distances.sort(key=lambda x: x[1])
            
            # Return the top 3 matches
            return [col for col, dist in distances[:3] if dist < len(column) / 2]
        except ImportError:
            # Fall back to simple substring matching if Levenshtein is not available
            return [col for col in available_columns 
                   if col.lower() in column.lower() or column.lower() in col.lower()]