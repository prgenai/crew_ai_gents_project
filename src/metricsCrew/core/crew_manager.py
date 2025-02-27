"""
Crew Manager Module

This module manages the CrewAI crew and orchestrates the workflow between agents.
"""

from typing import Dict, Any, Optional, List
from crewai import Crew, Process
import os

from src.metricsCrew.agents.manager_agent import ManagerAgent
from src.metricsCrew.agents.schema_agent import SchemaAgent
from src.metricsCrew.agents.query_agent import QueryAgent
from src.metricsCrew.agents.code_agent import CodeAgent
from src.metricsCrew.agents.result_agent import ResultAgent
from src.metricsCrew.tasks.task_definitions import (
    create_schema_analysis_task,
    create_query_interpretation_task,
    create_code_generation_task,
    create_result_analysis_task,
    create_response_synthesis_task
)
from src.metricsCrew.core.data_manager import DataManager
from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.config import get_config, get_data_source_config


class CrewManager:
    """
    Manages the CrewAI crew and orchestrates the workflow between agents.
    
    This class initializes all agents, creates tasks, and manages the execution
    of the business data analysis workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CrewManager.
        
        Args:
            config: Optional configuration dictionary
        """
        # Load configuration
        self.config = config or get_config()
        
        # Check for Google API key
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Initialize shared components
        self.data_manager = DataManager(cache_enabled=True)
        self.schema_registry = SchemaRegistry()
        
        # Initialize agents
        self.manager_agent = ManagerAgent(
            config=get_config('agents.manager_agent')
        ).get_agent()
        
        self.schema_agent = SchemaAgent(
            config=get_config('agents.schema_agent'),
            data_manager=self.data_manager,
            schema_registry=self.schema_registry
        ).get_agent()
        
        self.query_agent = QueryAgent(
            config=get_config('agents.query_agent'),
            schema_registry=self.schema_registry
        ).get_agent()
        
        self.code_agent = CodeAgent(
            config=get_config('agents.code_agent'),
            schema_registry=self.schema_registry
        ).get_agent()
        
        self.result_agent = ResultAgent(
            config=get_config('agents.result_agent'),
            data_manager=self.data_manager,
            schema_registry=self.schema_registry
        ).get_agent()
        
        # Initialize dataset mapping
        self.datasets = {}
        self._init_datasets()
    
    def _init_datasets(self):
        """Initialize and load configured datasets."""
        # Get data source configurations
        data_sources = get_data_source_config()
        
        if not data_sources:
            return
        
        for source_id, source_config in data_sources.items():
            if "path" in source_config:
                # Load the dataset
                dataset_path = source_config["path"]
                description = source_config.get("description", "")
                
                try:
                    # Load the dataset using the data manager
                    dataset_id = self.data_manager.load_dataset(dataset_path, source_id)
                    
                    # Register the schema
                    df = self.data_manager.get_dataframe(dataset_id)
                    self.schema_registry.register_schema(dataset_id, df, description)
                    
                    # Store the mapping
                    self.datasets[source_id] = {
                        "id": dataset_id,
                        "path": dataset_path,
                        "description": description
                    }
                    
                    print(f"Loaded dataset '{source_id}' from {dataset_path}")
                    
                except Exception as e:
                    print(f"Error loading dataset '{source_id}': {str(e)}")
    
    def analyze_query(self, query: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a business query using the CrewAI workflow.
        
        Args:
            query: Natural language business query
            dataset_id: Optional identifier for the dataset to query.
                       If None, the first available dataset will be used.
                       
        Returns:
            Dictionary with analysis results
        """
        # Determine the dataset to use
        if dataset_id is None:
            if not self.datasets:
                raise ValueError("No datasets available for analysis")
            dataset_id = next(iter(self.datasets.values()))["id"]
        elif dataset_id in self.datasets:
            dataset_id = self.datasets[dataset_id]["id"]
        
        try:
            # Get schema information
            schema_info = self.schema_registry.get_schema(dataset_id)
            
            # Create tasks
            query_task = create_query_interpretation_task(
                agent=self.query_agent,
                query=query,
                schema_info=schema_info
            )
            
            code_task = create_code_generation_task(
                agent=self.code_agent,
                requirements={},  # Will be populated from query_task
                schema_info=schema_info
            )
            
            result_task = create_result_analysis_task(
                agent=self.result_agent,
                code="",  # Will be populated from code_task
                requirements={},  # Will be populated from query_task
                dataset_id=dataset_id,
                query=query
            )
            
            response_task = create_response_synthesis_task(
                agent=self.manager_agent,
                query=query,
                schema_info=schema_info,
                query_interpretation={},  # Will be populated from query_task
                execution_results={},  # Will be populated from result_task
                business_interpretation={}  # Will be populated from result_task
            )
            
            # Create the crew
            crew = Crew(
                agents=[
                    self.manager_agent,
                    self.query_agent,
                    self.code_agent,
                    self.result_agent
                ],
                tasks=[
                    query_task,
                    code_task,
                    result_task,
                    response_task
                ],
                verbose=True,
                process=Process.sequential
            )
            
            # Execute the crew
            result = crew.kickoff(
                inputs={
                    "query": query,
                    "dataset_id": dataset_id,
                    "schema_info": schema_info
                }
            )
            
            return {
                "result": result,
                "query": query,
                "dataset_id": dataset_id
            }
            
        except Exception as e:
            return {"error": f"Error analyzing query: {str(e)}"}
    
    def analyze_dataset(self, dataset_path: str, dataset_id: Optional[str] = None,
                      description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a dataset's structure using the schema agent.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_id: Optional identifier for the dataset
            description: Optional description of the dataset
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Create schema analysis task
            schema_task = create_schema_analysis_task(
                agent=self.schema_agent,
                dataset_path=dataset_path,
                dataset_id=dataset_id,
                description=description
            )
            
            # Create a single-task crew
            crew = Crew(
                agents=[self.schema_agent],
                tasks=[schema_task],
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff(
                inputs={
                    "dataset_path": dataset_path,
                    "dataset_id": dataset_id,
                    "description": description
                }
            )
            
            # If successful, add to the datasets mapping
            if isinstance(result, dict) and "dataset_id" in result:
                actual_dataset_id = result["dataset_id"]
                self.datasets[actual_dataset_id] = {
                    "id": actual_dataset_id,
                    "path": dataset_path,
                    "description": description or ""
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Error analyzing dataset: {str(e)}"}
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary mapping dataset IDs to dataset information
        """
        return self.datasets