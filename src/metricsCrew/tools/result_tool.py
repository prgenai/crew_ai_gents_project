"""
Result Analysis Tool Module

This module provides a tool for executing analytical code and interpreting
the results in a business context.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import json
import traceback
import re

from src.metricsCrew.core.data_manager import DataManager
from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.config import get_tool_config


class ResultAnalysisTool(BaseTool):
    """
    Tool for executing analytical code and interpreting the results.
    
    This tool executes generated Python code against a dataset and interprets
    the results in a business context, providing insights and explanations.
    """
    
    name = "result_analysis_tool"
    description = "Executes analytical code and interprets results in business context"
    
    def __init__(self, data_manager: Optional[DataManager] = None, 
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the ResultAnalysisTool.
        
        Args:
            data_manager: DataManager instance for dataset access
            schema_registry: SchemaRegistry instance for metadata access
        """
        super().__init__()
        self.data_manager = data_manager or DataManager()
        self.schema_registry = schema_registry or SchemaRegistry()
        
        # Load tool configuration
        self.config = get_tool_config('result_analysis_tool', {})
        self.execution_timeout = self.config.get('execution_timeout', 30)  # seconds
    
    def _run(self, code: str, dataset_id: str, requirements: Dict[str, Any] = None,
            query: str = None) -> Dict[str, Any]:
        """
        Execute code and interpret the results.
        
        Args:
            code: Python code to execute
            dataset_id: Identifier for the dataset
            requirements: Optional analytical requirements dictionary
            query: Optional original user query
            
        Returns:
            Dictionary with execution results and business interpretation
        """
        # Execute the code
        execution_result = self.execute_code(code, dataset_id)
        
        # Check for execution errors
        if "error" in execution_result:
            return execution_result
        
        # If requirements and query are provided, interpret the results
        if requirements and query:
            interpretation = self.interpret_results(
                execution_result, requirements, query, dataset_id
            )
            
            return {
                "execution_results": execution_result,
                "business_interpretation": interpretation
            }
        
        # Otherwise, just return the execution results
        return execution_result
    
    def _arun(self, code: str, dataset_id: str, requirements: Dict[str, Any] = None,
             query: str = None):
        """Async implementation of the tool."""
        # This tool doesn't have an async implementation
        raise NotImplementedError("This tool does not support async")
    
    def execute_code(self, code: str, dataset_id: str) -> Dict[str, Any]:
        """
        Execute Python code against a dataset.
        
        Args:
            code: Python code to execute
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Ensure the dataset is loaded
            if not hasattr(self.data_manager, '_datasets') or dataset_id not in self.data_manager._datasets:
                return {"error": f"Dataset not found: {dataset_id}"}
            
            # Check if the code defines the analyze_data function
            if not re.search(r'def\s+analyze_data', code):
                return {"error": "Code does not define the analyze_data function"}
            
            # Prepare execution environment
            local_vars = {
                "pd": pd,
                "np": np,
                "df": self.data_manager.get_dataframe(dataset_id)
            }
            
            # Execute the code to define the analyze_data function
            exec(code, {"pd": pd, "np": np}, local_vars)
            
            # Check if analyze_data function is defined
            if "analyze_data" not in local_vars:
                return {"error": "analyze_data function not found in executed code"}
            
            # Call the analyze_data function
            analyze_data = local_vars["analyze_data"]
            result = analyze_data(self.data_manager.get_dataframe(dataset_id))
            
            # Process the result
            if isinstance(result, pd.DataFrame):
                # Convert DataFrame to records for JSON serialization
                processed_result = {
                    "type": "dataframe",
                    "shape": result.shape,
                    "columns": list(result.columns),
                    "records": result.head(50).to_dict(orient="records"),  # Limit to 50 rows
                    "summary": self._summarize_dataframe(result)
                }
            elif isinstance(result, dict):
                # Dictionary result
                processed_result = {
                    "type": "dictionary",
                    "keys": list(result.keys()),
                    "values": self._process_dict_values(result)
                }
            elif isinstance(result, (list, tuple)):
                # List or tuple result
                processed_result = {
                    "type": "list",
                    "length": len(result),
                    "elements": self._process_list_values(result)
                }
            else:
                # Other result types
                processed_result = {
                    "type": str(type(result).__name__),
                    "value": str(result)
                }
            
            return {
                "success": True,
                "result": processed_result
            }
            
        except Exception as e:
            # Get the full traceback
            tb = traceback.format_exc()
            
            return {
                "error": f"Error executing code: {str(e)}",
                "traceback": tb
            }
    
    def _summarize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a summary of a DataFrame.
        
        Args:
            df: Pandas DataFrame to summarize
            
        Returns:
            Dictionary with DataFrame summary
        """
        try:
            # Get basic statistics for numeric columns
            numeric_stats = df.describe().to_dict()
            
            # Count distinct values for non-numeric columns
            non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
            distinct_counts = {col: df[col].nunique() for col in non_numeric_columns}
            
            # Check for missing values
            missing_counts = df.isnull().sum().to_dict()
            
            return {
                "row_count": len(df),
                "numeric_stats": numeric_stats,
                "distinct_counts": distinct_counts,
                "missing_counts": missing_counts
            }
        except Exception as e:
            return {"error": f"Error summarizing DataFrame: {str(e)}"}
    
    def _process_dict_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process dictionary values for JSON serialization.
        
        Args:
            d: Dictionary to process
            
        Returns:
            Processed dictionary
        """
        processed = {}
        
        for key, value in d.items():
            if isinstance(value, pd.DataFrame):
                processed[key] = {
                    "type": "dataframe",
                    "shape": value.shape,
                    "preview": value.head(5).to_dict(orient="records")
                }
            elif isinstance(value, (pd.Series, np.ndarray)):
                processed[key] = {
                    "type": str(type(value).__name__),
                    "shape": value.shape,
                    "preview": value[:5].tolist() if len(value) > 0 else []
                }
            elif isinstance(value, (dict, list, tuple)):
                processed[key] = {
                    "type": str(type(value).__name__),
                    "preview": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                }
            else:
                processed[key] = {
                    "type": str(type(value).__name__),
                    "value": str(value)
                }
        
        return processed
    
    def _process_list_values(self, lst: List[Any]) -> List[Any]:
        """
        Process list values for JSON serialization.
        
        Args:
            lst: List to process
            
        Returns:
            Processed list
        """
        # Limit to first 20 elements
        preview_list = lst[:20]
        processed = []
        
        for item in preview_list:
            if isinstance(item, pd.DataFrame):
                processed.append({
                    "type": "dataframe",
                    "shape": item.shape,
                    "preview": item.head(3).to_dict(orient="records")
                })
            elif isinstance(item, (pd.Series, np.ndarray)):
                processed.append({
                    "type": str(type(item).__name__),
                    "shape": item.shape,
                    "preview": item[:3].tolist() if len(item) > 0 else []
                })
            elif isinstance(item, dict):
                processed.append({
                    "type": "dictionary",
                    "keys": list(item.keys()),
                    "preview": str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                })
            else:
                processed.append({
                    "type": str(type(item).__name__),
                    "value": str(item)
                })
        
        if len(lst) > 20:
            processed.append({"note": f"...{len(lst) - 20} more items not shown"})
        
        return processed
    
    def interpret_results(self, results: Dict[str, Any], requirements: Dict[str, Any],
                         query: str, dataset_id: str) -> Dict[str, Any]:
        """
        Interpret execution results in a business context.
        
        Args:
            results: Execution results from execute_code
            requirements: Analytical requirements dictionary
            query: Original user query
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary with business interpretation
        """
        try:
            # Get schema information
            schema = self.schema_registry.get_schema(dataset_id)
            
            # Prepare a prompt for the LLM
            from langchain_google_genai import ChatGoogleGenerativeAI
            import os
            
            # Get LLM configuration
            model_name = "gemini-pro"
            temperature = 0.4
            
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
            
            # Create the prompt for result interpretation
            prompt = f"""You are a Business Intelligence Analyst specializing in interpreting data analysis results.
            Your task is to explain the following analysis results in business-friendly terms.
            
            Original User Query:
            "{query}"
            
            Analysis Requirements:
            {json.dumps(requirements, indent=2)}
            
            Analysis Results:
            {json.dumps(results.get('result', {}), indent=2)}
            
            Dataset Information:
            {json.dumps({
                "metrics": self.schema_registry.get_metrics(dataset_id),
                "dimensions": self.schema_registry.get_dimensions(dataset_id),
                "timestamps": self.schema_registry.get_timestamps(dataset_id)
            }, indent=2)}
            
            Provide a comprehensive business interpretation of these results that:
            1. Summarizes the key findings in business language
            2. Explains what the data shows and why it matters
            3. Highlights any notable patterns, trends, or anomalies
            4. Suggests potential business implications or next steps
            5. Translates technical metrics into business value
            
            Format your response as a JSON with these sections:
            {
              "summary": "A brief executive summary of key findings",
              "detailed_insights": ["Insight 1", "Insight 2", ...],
              "business_implications": ["Implication 1", "Implication 2", ...],
              "recommendations": ["Recommendation 1", "Recommendation 2", ...],
              "additional_analysis": "Suggestions for further analysis if applicable"
            }
            """
            
            # Get the response from the LLM
            response = llm.invoke(prompt)
            
            # Extract the JSON response
            try:
                import re
                
                # Extract JSON from the response
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                
                if json_match:
                    interpretation = json.loads(json_match.group(1))
                    return interpretation
                else:
                    # If no JSON is found, create a structured interpretation from the text
                    text = response.content.strip()
                    
                    # Create a simple structure
                    return {
                        "summary": "Analysis results interpretation",
                        "detailed_insights": [text],
                        "business_implications": [],
                        "recommendations": [],
                        "additional_analysis": ""
                    }
                    
            except Exception as e:
                return {
                    "error": f"Error parsing interpretation: {str(e)}",
                    "raw_response": response.content
                }
            
        except Exception as e:
            return {"error": f"Error interpreting results: {str(e)}"}