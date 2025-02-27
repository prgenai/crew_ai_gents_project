"""
Schema Registry Module

This module provides functionality for storing and accessing metadata about datasets,
including column information, relationships, and business term mappings.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import json


class SchemaRegistry:
    """
    Stores and manages metadata about datasets.
    
    The SchemaRegistry maintains information about dataset structure, relationships
    between fields, and mappings between business terminology and technical fields.
    This enables agents to understand and work with datasets at a semantic level.
    """
    
    def __init__(self):
        """Initialize the SchemaRegistry."""
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._business_glossary: Dict[str, str] = {}
        self._relationships: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_schema(self, dataset_id: str, df: pd.DataFrame, 
                       description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze and register schema information for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            df: Pandas DataFrame containing the dataset
            description: Optional description of the dataset
            
        Returns:
            Dictionary containing the registered schema information
        """
        # Extract basic column information
        columns_info = {}
        
        for column in df.columns:
            dtype = df[column].dtype
            sample_values = df[column].dropna().head(5).tolist()
            unique_count = df[column].nunique()
            is_unique = unique_count == len(df)
            
            # Determine if column might be a key
            is_potential_key = is_unique or (unique_count > 0.9 * len(df))
            
            # Infer column category (dimension, metric, timestamp)
            category = self._infer_column_category(df[column], column)
            
            columns_info[column] = {
                "data_type": str(dtype),
                "python_type": df[column].dtype.type.__name__,
                "sample_values": sample_values,
                "unique_values": unique_count,
                "is_potential_key": is_potential_key,
                "category": category,
                "description": "",  # To be filled later
                "business_terms": [],  # To be filled later
                "has_missing_values": df[column].isna().any(),
                "missing_count": df[column].isna().sum()
            }
        
        # Store schema information
        schema = {
            "dataset_id": dataset_id,
            "description": description or "",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns_info
        }
        
        self._schemas[dataset_id] = schema
        self._relationships[dataset_id] = []
        
        # Try to infer relationships between columns
        self._infer_relationships(dataset_id, df)
        
        # Generate initial business glossary entries
        self._generate_initial_business_terms(dataset_id)
        
        return schema
    
    def _infer_column_category(self, series: pd.Series, column_name: str) -> str:
        """
        Infer the category of a column (dimension, metric, timestamp).
        
        Args:
            series: The pandas Series to analyze
            column_name: The name of the column
            
        Returns:
            Category as a string: "dimension", "metric", or "timestamp"
        """
        # Check if it's a timestamp
        if pd.api.types.is_datetime64_any_dtype(series):
            return "timestamp"
        
        # Check if it's a numeric type that might be a metric
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # Numeric columns with many distinct values likely represent metrics
            # Names containing common metric terms are likely metrics
            metric_terms = ["amount", "sum", "total", "count", "price", "cost", 
                           "profit", "revenue", "sales", "quantity", "qty", "number"]
            
            if any(term in column_name.lower() for term in metric_terms):
                return "metric"
                
            # If it has decimal places or many unique values, likely a metric
            if series.dtype == float or series.nunique() > 100:
                return "metric"
        
        # Default to dimension for categorical, boolean, or string columns
        return "dimension"
    
    def _infer_relationships(self, dataset_id: str, df: pd.DataFrame):
        """
        Infer relationships between columns in the dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            df: Pandas DataFrame containing the dataset
        """
        columns = list(df.columns)
        
        # Look for columns with similar names that might be related
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Check for ID relationships (e.g., customer_id and customer_name)
                if col1.endswith('_id') and col2.startswith(col1[:-3]):
                    self._add_relationship(dataset_id, col1, col2, "one-to-many")
                elif col2.endswith('_id') and col1.startswith(col2[:-3]):
                    self._add_relationship(dataset_id, col2, col1, "one-to-many")
                
                # Check for parent-child relationships in names (Category, Sub-Category)
                if col1.lower() in ['category', 'type', 'group'] and col2.lower().startswith('sub'):
                    self._add_relationship(dataset_id, col1, col2, "parent-child")
        
        # Look for columns with similar values that might be related
        numeric_columns = df.select_dtypes(include=['number']).columns
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                # If there's a consistent mathematical relationship, add it
                if all(df[col1] >= df[col2]) and any(df[col1] > df[col2]):
                    self._add_relationship(dataset_id, col1, col2, "greater-than")
    
    def _add_relationship(self, dataset_id: str, from_column: str, to_column: str, 
                         relationship_type: str):
        """
        Add a relationship between two columns.
        
        Args:
            dataset_id: Identifier for the dataset
            from_column: Source column name
            to_column: Target column name
            relationship_type: Type of relationship
        """
        self._relationships[dataset_id].append({
            "from_column": from_column,
            "to_column": to_column,
            "relationship_type": relationship_type
        })
    
    def _generate_initial_business_terms(self, dataset_id: str):
        """
        Generate initial business glossary entries from column names.
        
        Args:
            dataset_id: Identifier for the dataset
        """
        schema = self._schemas.get(dataset_id)
        if not schema:
            return
        
        for column, info in schema["columns"].items():
            # Convert snake_case or CamelCase to readable form
            readable_name = self._make_readable_name(column)
            
            # Add to business glossary
            self._business_glossary[readable_name] = column
            
            # Add to column's business terms
            info["business_terms"].append(readable_name)
            
            # Add specific terms for known metric categories
            if info["category"] == "metric":
                if "sales" in column.lower():
                    info["business_terms"].extend(["revenue", "sales amount"])
                elif "profit" in column.lower():
                    info["business_terms"].extend(["margin", "profit amount"])
                elif "quantity" in column.lower() or "qty" in column.lower():
                    info["business_terms"].extend(["volume", "quantity sold"])
    
    def _make_readable_name(self, column_name: str) -> str:
        """
        Convert a technical column name to a readable business term.
        
        Args:
            column_name: Technical column name
            
        Returns:
            Readable business term
        """
        # Replace underscores with spaces
        readable = column_name.replace('_', ' ')
        
        # Handle CamelCase
        import re
        readable = re.sub(r'(?<!^)(?=[A-Z])', ' ', readable)
        
        # Capitalize first letter of each word
        readable = readable.title()
        
        return readable
    
    def get_schema(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get the schema information for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary containing schema information
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        if dataset_id not in self._schemas:
            raise KeyError(f"Schema not found for dataset: {dataset_id}")
        
        return self._schemas[dataset_id]
    
    def get_column_info(self, dataset_id: str, column_name: str) -> Dict[str, Any]:
        """
        Get information about a specific column.
        
        Args:
            dataset_id: Identifier for the dataset
            column_name: Name of the column
            
        Returns:
            Dictionary containing column information
            
        Raises:
            KeyError: If the dataset_id or column is not found
        """
        schema = self.get_schema(dataset_id)
        
        if column_name not in schema["columns"]:
            raise KeyError(f"Column not found: {column_name}")
        
        return schema["columns"][column_name]
    
    def get_business_term_mapping(self, term: str) -> List[Tuple[str, str]]:
        """
        Find technical fields that match a business term.
        
        Args:
            term: Business term to search for
            
        Returns:
            List of (dataset_id, column_name) tuples that match the term
        """
        term = term.lower()
        matches = []
        
        # Check exact match in business glossary
        if term in [t.lower() for t in self._business_glossary]:
            for business_term, column in self._business_glossary.items():
                if business_term.lower() == term:
                    # Find which dataset(s) contain this column
                    for dataset_id, schema in self._schemas.items():
                        if column in schema["columns"]:
                            matches.append((dataset_id, column))
        
        # Check for partial matches in column business terms
        for dataset_id, schema in self._schemas.items():
            for column, info in schema["columns"].items():
                for business_term in info.get("business_terms", []):
                    if term in business_term.lower():
                        matches.append((dataset_id, column))
        
        return list(set(matches))  # Remove duplicates
    
    def update_column_description(self, dataset_id: str, column_name: str, 
                                 description: str):
        """
        Update the description for a column.
        
        Args:
            dataset_id: Identifier for the dataset
            column_name: Name of the column
            description: New description
            
        Raises:
            KeyError: If the dataset_id or column is not found
        """
        column_info = self.get_column_info(dataset_id, column_name)
        column_info["description"] = description
    
    def add_business_term(self, term: str, technical_field: str):
        """
        Add or update a business term in the glossary.
        
        Args:
            term: Business term
            technical_field: Corresponding technical field name
        """
        self._business_glossary[term] = technical_field
        
        # Add to any column that matches this technical field
        for dataset_id, schema in self._schemas.items():
            for column, info in schema["columns"].items():
                if column == technical_field:
                    if term not in info["business_terms"]:
                        info["business_terms"].append(term)
    
    def export_schema(self, dataset_id: str, format: str = "json") -> str:
        """
        Export the schema information in a specified format.
        
        Args:
            dataset_id: Identifier for the dataset
            format: Export format (currently only "json" is supported)
            
        Returns:
            Schema information in the specified format
            
        Raises:
            KeyError: If the dataset_id is not found
            ValueError: If the format is not supported
        """
        schema = self.get_schema(dataset_id)
        
        if format.lower() == "json":
            return json.dumps(schema, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_metrics(self, dataset_id: str) -> List[str]:
        """
        Get a list of metric columns for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            List of column names that are categorized as metrics
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        schema = self.get_schema(dataset_id)
        
        return [column for column, info in schema["columns"].items() 
                if info["category"] == "metric"]
    
    def get_dimensions(self, dataset_id: str) -> List[str]:
        """
        Get a list of dimension columns for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            List of column names that are categorized as dimensions
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        schema = self.get_schema(dataset_id)
        
        return [column for column, info in schema["columns"].items() 
                if info["category"] == "dimension"]
    
    def get_timestamps(self, dataset_id: str) -> List[str]:
        """
        Get a list of timestamp columns for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            List of column names that are categorized as timestamps
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        schema = self.get_schema(dataset_id)
        
        return [column for column, info in schema["columns"].items() 
                if info["category"] == "timestamp"]
    
    def get_relationships(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            List of relationship dictionaries
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        if dataset_id not in self._relationships:
            raise KeyError(f"Relationships not found for dataset: {dataset_id}")
        
        return self._relationships[dataset_id]