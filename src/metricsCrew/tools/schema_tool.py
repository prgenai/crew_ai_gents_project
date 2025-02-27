"""
Schema Analysis Tool Module

This module provides a tool for analyzing dataset structure and creating metadata.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Optional
import pandas as pd
import json
import os

from src.metricsCrew.core.data_manager import DataManager
from src.metricsCrew.core.schema_registry import SchemaRegistry


class SchemaAnalysisTool(BaseTool):
    """
    Tool for analyzing dataset structure and creating metadata.
    
    This tool uses the DataManager to load a dataset and the SchemaRegistry
    to analyze and store metadata about the dataset structure.
    """
    
    name = "schema_analysis_tool"
    description = "Analyzes a dataset's structure and creates metadata about columns, relationships, and business terminology"
    
    def __init__(self, data_manager: Optional[DataManager] = None, 
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the SchemaAnalysisTool.
        
        Args:
            data_manager: DataManager instance for loading datasets
            schema_registry: SchemaRegistry instance for storing metadata
        """
        super().__init__()
        self.data_manager = data_manager or DataManager()
        self.schema_registry = schema_registry or SchemaRegistry()
    
    def _run(self, dataset_path: str, dataset_id: Optional[str] = None,
            description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a dataset and create metadata.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_id: Optional identifier for the dataset
            description: Optional description of the dataset
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            ValueError: If there is an error analyzing the dataset
        """
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                return {"error": f"Dataset file not found: {dataset_path}"}
            
            # Load the dataset
            actual_dataset_id = self.data_manager.load_dataset(dataset_path, dataset_id)
            
            # Get the dataframe
            df = self.data_manager.get_dataframe(actual_dataset_id)
            
            # Register the schema
            schema = self.schema_registry.register_schema(
                actual_dataset_id, df, description
            )
            
            # Get column categories
            metrics = self.schema_registry.get_metrics(actual_dataset_id)
            dimensions = self.schema_registry.get_dimensions(actual_dataset_id)
            timestamps = self.schema_registry.get_timestamps(actual_dataset_id)
            
            # Get relationships
            relationships = self.schema_registry.get_relationships(actual_dataset_id)
            
            # Prepare analysis results
            result = {
                "dataset_id": actual_dataset_id,
                "file_path": dataset_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_list": list(df.columns),
                "metrics": metrics,
                "dimensions": dimensions,
                "timestamps": timestamps,
                "relationships": relationships,
                "sample_data": df.head(5).to_dict(orient="records"),
                "description": description or "",
                "full_schema": schema
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error analyzing dataset: {str(e)}"}
    
    def _arun(self, dataset_path: str, dataset_id: Optional[str] = None,
             description: Optional[str] = None):
        """Async implementation of the tool."""
        # This tool doesn't have an async implementation
        raise NotImplementedError("This tool does not support async")
    
    def get_schema_json(self, dataset_id: str) -> str:
        """
        Get the schema information in JSON format.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            JSON string containing schema information
        """
        try:
            schema = self.schema_registry.get_schema(dataset_id)
            return json.dumps(schema, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    def get_business_glossary(self) -> Dict[str, str]:
        """
        Get the business glossary mapping.
        
        Returns:
            Dictionary mapping business terms to technical field names
        """
        return self.schema_registry._business_glossary
    
    def get_column_metadata(self, dataset_id: str, column_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific column.
        
        Args:
            dataset_id: Identifier for the dataset
            column_name: Name of the column
            
        Returns:
            Dictionary containing column metadata
        """
        try:
            return self.schema_registry.get_column_info(dataset_id, column_name)
        except Exception as e:
            return {"error": str(e)}