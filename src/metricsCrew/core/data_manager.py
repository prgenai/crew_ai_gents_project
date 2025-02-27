"""
Data Manager Module

This module provides functionality for loading, accessing, and managing datasets
used in the Business Data Analysis System.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Any


class DataManager:
    """
    Manages dataset loading and access for the analysis system.
    
    This class handles loading datasets from various file formats,
    provides caching for performance, and offers standard interfaces
    for accessing and manipulating data.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the DataManager.
        
        Args:
            cache_enabled: Whether to cache loaded datasets in memory
        """
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._cache_enabled = cache_enabled
        
    def load_dataset(self, file_path: str, dataset_id: Optional[str] = None, **kwargs) -> str:
        """
        Load a dataset from a file.
        
        Args:
            file_path: Path to the dataset file
            dataset_id: Optional identifier for the dataset. If not provided,
                        the file name without extension will be used
            **kwargs: Additional arguments to pass to the loading function
                      (e.g., pandas.read_csv parameters)
        
        Returns:
            The dataset_id that can be used to access the loaded dataset
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Use filename as dataset_id if not provided
        if dataset_id is None:
            dataset_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Determine file format and load accordingly
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                # Set default parameters for CSV loading
                csv_params = {
                    'low_memory': False,
                    'encoding': 'utf-8'
                }
                csv_params.update(kwargs)
                df = pd.read_csv(file_path, **csv_params)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
            # Store in cache if enabled
            if self._cache_enabled:
                self._datasets[dataset_id] = df
                
            return dataset_id
        
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """
        Get a dataset as a pandas DataFrame.
        
        Args:
            dataset_id: Identifier of the dataset to retrieve
            
        Returns:
            The dataset as a pandas DataFrame
            
        Raises:
            KeyError: If the dataset_id is not found
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        
        return self._datasets[dataset_id]
    
    def execute_code(self, code: str, dataset_id: str) -> Dict[str, Any]:
        """
        Execute Python code on a dataset.
        
        Args:
            code: Python code to execute
            dataset_id: Dataset to use as input
            
        Returns:
            Dictionary of variables created during execution
            
        Raises:
            KeyError: If the dataset_id is not found
            Exception: If there is an error executing the code
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        
        # Prepare execution environment
        df = self._datasets[dataset_id]
        local_vars = {"df": df, "pd": pd, "np": np}
        
        try:
            # Execute the code
            exec(code, {"pd": pd, "np": np}, local_vars)
            
            # Filter out built-in variables
            result = {k: v for k, v in local_vars.items() 
                     if not k.startswith("__") and k != "df"}
            
            return result
        except Exception as e:
            raise Exception(f"Error executing code: {str(e)}")
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_id: Identifier of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        
        df = self._datasets[dataset_id]
        
        # Compile basic dataset information
        info = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "has_missing_values": df.isna().any().any()
        }
        
        return info
    
    def clear_cache(self, dataset_id: Optional[str] = None):
        """
        Clear cached datasets.
        
        Args:
            dataset_id: If provided, only clear this specific dataset.
                        If None, clear all cached datasets.
        """
        if dataset_id is None:
            self._datasets.clear()
        elif dataset_id in self._datasets:
            del self._datasets[dataset_id]