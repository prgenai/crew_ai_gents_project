"""
Schema Analysis Agent Module

This module implements a CrewAI agent specialized in analyzing dataset structure
and creating metadata about datasets.
"""

import os
from typing import Dict, Any, Optional
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.metricsCrew.tools.schema_tool import SchemaAnalysisTool
from src.metricsCrew.core.data_manager import DataManager
from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.config import get_agent_config, get_prompt_template


class SchemaAgent:
    """
    Agent specialized in analyzing dataset structure and relationships.
    
    This agent uses the SchemaAnalysisTool to examine datasets and extract
    metadata about their structure, including column information, data types,
    relationships between columns, and business terminology mappings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                data_manager: Optional[DataManager] = None,
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the SchemaAgent.
        
        Args:
            config: Configuration dictionary for the agent
            data_manager: DataManager instance for dataset access
            schema_registry: SchemaRegistry instance for storing metadata
        """
        # Load configuration with defaults
        self.config = config or get_agent_config('schema_agent', {})
        
        # Set up dependencies
        self.data_manager = data_manager or DataManager()
        self.schema_registry = schema_registry or SchemaRegistry()
        
        # Create the schema analysis tool
        self.schema_tool = SchemaAnalysisTool(
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
        temperature = llm_config.get('temperature', 0.1)
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Get prompt template for enhancing schema analysis
        schema_prompt = get_prompt_template('schema_agent.prompt', 
            """You are a Data Architect specialized in understanding dataset structure and creating metadata.
            Your task is to analyze datasets and provide comprehensive information about their structure.
            - Identify the purpose of each column
            - Classify columns as dimensions, metrics, or timestamps
            - Detect relationships between columns
            - Create business-friendly descriptions
            - Map technical field names to business terminology
            
            When analyzing a dataset, be thorough and precise in your assessment.""")
        
        # Create the agent
        agent = Agent(
            role=self.config.get('role', "Data Architect"),
            goal=self.config.get('goal', "Analyze dataset structure and provide data context"),
            backstory=self.config.get('backstory', 
                "An experienced data specialist who can quickly understand database schemas and business data models"),
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', False),
            llm=llm,
            tools=[self.schema_tool]
        )
        
        return agent
    
    def analyze_dataset(self, dataset_path: str, dataset_id: Optional[str] = None,
                       description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a dataset and create metadata.
        
        This is a convenience method that directly uses the schema tool
        without going through the CrewAI task framework.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_id: Optional identifier for the dataset
            description: Optional description of the dataset
            
        Returns:
            Dictionary containing analysis results
        """
        return self.schema_tool._run(
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            description=description
        )
    
    def enhance_schema_descriptions(self, dataset_id: str) -> Dict[str, Any]:
        """
        Use the LLM to enhance schema descriptions with business context.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            Dictionary with enhanced schema information
            
        Note:
            This requires that the dataset has already been analyzed
            with analyze_dataset() or through a CrewAI task.
        """
        try:
            # Get the current schema
            schema = self.schema_registry.get_schema(dataset_id)
            
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
            
            # Get sample data
            df = self.data_manager.get_dataframe(dataset_id)
            sample_data = df.head(5).to_dict(orient="records")
            
            # For each column, generate an enhanced description
            for column_name, column_info in schema['columns'].items():
                # Skip if there's already a detailed description
                if len(column_info.get('description', '')) > 50:
                    continue
                
                # Get sample values for this column
                sample_values = [str(record.get(column_name, '')) for record in sample_data]
                
                # Create a prompt for the LLM
                prompt = f"""You are a Data Architect analyzing a dataset column.
                
                Column Name: {column_name}
                Data Type: {column_info['data_type']}
                Category: {column_info['category']}
                Sample Values: {', '.join(sample_values[:5])}
                
                Based on this information, please provide:
                1. A clear business description of what this column represents
                2. Additional business terms that might refer to this data
                3. Any business rules or constraints that might apply
                
                Format your response as a JSON with these keys: "description", "business_terms", "business_rules"
                """
                
                # Get the response from the LLM
                response = llm.invoke(prompt)
                
                # Extract the JSON response (handle potential formatting issues)
                try:
                    import json
                    import re
                    
                    # Extract JSON from the response
                    json_match = re.search(r'({.*})', response.content, re.DOTALL)
                    
                    if json_match:
                        enhancement = json.loads(json_match.group(1))
                        
                        # Update the schema registry
                        if 'description' in enhancement:
                            self.schema_registry.update_column_description(
                                dataset_id, column_name, enhancement['description']
                            )
                        
                        # Add business terms
                        if 'business_terms' in enhancement and isinstance(enhancement['business_terms'], list):
                            for term in enhancement['business_terms']:
                                if term and isinstance(term, str):
                                    self.schema_registry.add_business_term(term, column_name)
                except Exception as e:
                    print(f"Error parsing LLM response for column {column_name}: {str(e)}")
            
            # Return the enhanced schema
            return self.schema_registry.get_schema(dataset_id)
            
        except Exception as e:
            return {"error": f"Error enhancing schema: {str(e)}"}
    
    def get_dataset_summary(self, dataset_id: str) -> str:
        """
        Generate a human-readable summary of the dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            String containing a summary of the dataset
        """
        try:
            # Get the schema
            schema = self.schema_registry.get_schema(dataset_id)
            
            # Get metrics, dimensions, and timestamps
            metrics = self.schema_registry.get_metrics(dataset_id)
            dimensions = self.schema_registry.get_dimensions(dataset_id)
            timestamps = self.schema_registry.get_timestamps(dataset_id)
            
            # Get relationships
            relationships = self.schema_registry.get_relationships(dataset_id)
            
            # Create summary
            summary = [
                f"# Dataset Summary: {dataset_id}",
                f"\nThis dataset contains {schema['row_count']} records with {schema['column_count']} columns.",
                "\n## Key Metrics:",
                ", ".join(metrics) if metrics else "None identified",
                "\n## Dimensions for Analysis:",
                ", ".join(dimensions) if dimensions else "None identified",
                "\n## Time Dimensions:",
                ", ".join(timestamps) if timestamps else "None identified",
            ]
            
            if relationships:
                summary.append("\n## Identified Relationships:")
                for rel in relationships:
                    summary.append(f"- {rel['from_column']} → {rel['to_column']} ({rel['relationship_type']})")
            
            # Add column details
            summary.append("\n## Column Details:")
            for column, info in schema['columns'].items():
                desc = info.get('description', 'No description available')
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                
                summary.append(f"\n### {column} ({info['category']})")
                summary.append(f"- Type: {info['data_type']}")
                summary.append(f"- Description: {desc}")
                
                if info.get('business_terms'):
                    summary.append(f"- Business Terms: {', '.join(info['business_terms'])}")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error generating dataset summary: {str(e)}"