"""
Configuration Loader Module

This module provides functionality for loading and accessing configuration settings
from YAML files, with support for environment variable substitution.
"""

import os
import yaml
import re
from typing import Dict, Any, Optional, List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pattern for environment variable substitution ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r'\${([A-Za-z0-9_]+)}')

class ConfigLoader:
    """
    Loads and manages configuration settings from YAML files.
    
    This class handles loading configuration from YAML files, substituting
    environment variables, and providing access to configuration values.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_dir: Directory containing configuration files.
                       If None, uses the 'config' directory in the same location as this file.
        """
        # Determine configuration directory
        if config_dir is None:
            self.config_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.config_dir = config_dir
            
        # Initialize configuration dictionaries
        self.config = {}
        self.agents_config = {}
        self.tasks_config = {}
        self.tools_config = {}
        self.prompts_config = {}
        self.data_sources_config = {}
        
        # Load all configuration files
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files."""
        # Main configuration
        self.config = self._load_yaml_file('config.yaml')
        
        # Specialized configurations
        self.agents_config = self._load_yaml_file('agents.yaml')
        self.tasks_config = self._load_yaml_file('tasks.yaml')
        self.tools_config = self._load_yaml_file('tools.yaml')
        self.prompts_config = self._load_yaml_file('prompts.yaml')
        self.data_sources_config = self._load_yaml_file('data_sources.yaml')
        
        logger.info("All configuration files loaded successfully")
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Dictionary containing the configuration from the YAML file,
            or an empty dictionary if the file doesn't exist
        """
        file_path = os.path.join(self.config_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Substitute environment variables
            if config:
                config = self._substitute_env_vars(config)
                
            return config or {}
        
        except Exception as e:
            logger.error(f"Error loading configuration file {filename}: {str(e)}")
            return {}
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Args:
            config: Configuration object (dict, list, or scalar value)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        
        elif isinstance(config, str):
            # Replace ${VAR_NAME} with environment variable value
            def replace_env_var(match):
                env_var_name = match.group(1)
                env_var_value = os.environ.get(env_var_name)
                
                if env_var_value is None:
                    logger.warning(f"Environment variable not found: {env_var_name}")
                    return match.group(0)  # Return the original ${VAR_NAME}
                
                return env_var_value
            
            return ENV_VAR_PATTERN.sub(replace_env_var, config)
        
        else:
            return config
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value from the main config.
        
        Args:
            key: Dot-separated path to the configuration value.
                 If None, returns the entire configuration.
            default: Default value to return if the key is not found
            
        Returns:
            Configuration value at the specified key, or the default value
        """
        if key is None:
            return self.config
            
        return self._get_nested_config(self.config, key, default)
    
    def get_agent_config(self, agent_name: Optional[str] = None, default: Any = None) -> Any:
        """
        Get agent configuration.
        
        Args:
            agent_name: Name of the agent. If None, returns all agent configurations.
            default: Default value to return if the agent is not found
            
        Returns:
            Configuration for the specified agent, or the default value
        """
        if agent_name is None:
            return self.agents_config
            
        return self.agents_config.get(agent_name, default)
    
    def get_task_config(self, task_name: Optional[str] = None, default: Any = None) -> Any:
        """
        Get task configuration.
        
        Args:
            task_name: Name of the task. If None, returns all task configurations.
            default: Default value to return if the task is not found
            
        Returns:
            Configuration for the specified task, or the default value
        """
        if task_name is None:
            return self.tasks_config
            
        return self.tasks_config.get(task_name, default)
    
    def get_tool_config(self, tool_name: Optional[str] = None, default: Any = None) -> Any:
        """
        Get tool configuration.
        
        Args:
            tool_name: Name of the tool. If None, returns all tool configurations.
            default: Default value to return if the tool is not found
            
        Returns:
            Configuration for the specified tool, or the default value
        """
        if tool_name is None:
            return self.tools_config
            
        return self.tools_config.get(tool_name, default)
    
    def get_prompt_template(self, template_name: str, default: str = "") -> str:
        """
        Get a prompt template.
        
        Args:
            template_name: Name of the prompt template
            default: Default template to return if not found
            
        Returns:
            The prompt template string
        """
        if template_name in self.prompts_config:
            return self.prompts_config[template_name]
        
        # Check for dotted notation (e.g., "agents.business_agent.prompt")
        return self._get_nested_config(self.prompts_config, template_name, default)
    
    def get_data_source_config(self, source_name: Optional[str] = None, default: Any = None) -> Any:
        """
        Get data source configuration.
        
        Args:
            source_name: Name of the data source. If None, returns all data source configurations.
            default: Default value to return if the data source is not found
            
        Returns:
            Configuration for the specified data source, or the default value
        """
        if source_name is None:
            return self.data_sources_config
            
        return self.data_sources_config.get(source_name, default)
    
    def _get_nested_config(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the configuration value (e.g., "logging.level")
            default: Default value to return if the key is not found
            
        Returns:
            Configuration value at the specified key, or the default value
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict):
                return default
                
            if key not in current:
                return default
                
            current = current[key]
            
        return current
    
    def reload_configs(self):
        """Reload all configuration files."""
        self._load_all_configs()


# Create a singleton instance of the config loader
config_loader = ConfigLoader()

# Convenience functions for accessing configuration
def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """Get a value from the main configuration."""
    return config_loader.get_config(key, default)

def get_agent_config(agent_name: Optional[str] = None, default: Any = None) -> Any:
    """Get agent configuration."""
    return config_loader.get_agent_config(agent_name, default)

def get_task_config(task_name: Optional[str] = None, default: Any = None) -> Any:
    """Get task configuration."""
    return config_loader.get_task_config(task_name, default)

def get_tool_config(tool_name: Optional[str] = None, default: Any = None) -> Any:
    """Get tool configuration."""
    return config_loader.get_tool_config(tool_name, default)

def get_prompt_template(template_name: str, default: str = "") -> str:
    """Get a prompt template."""
    return config_loader.get_prompt_template(template_name, default)

def get_data_source_config(source_name: Optional[str] = None, default: Any = None) -> Any:
    """Get data source configuration."""
    return config_loader.get_data_source_config(source_name, default)

def reload_configs():
    """Reload all configuration files."""
    config_loader.reload_configs()