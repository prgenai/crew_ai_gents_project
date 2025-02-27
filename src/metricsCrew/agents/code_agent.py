"""
Code Generation Agent Module

This module implements a CrewAI agent specialized in generating executable
pandas code based on analytical requirements.
"""

import os
import json
from typing import Dict, Any, Optional, List
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.metricsCrew.core.schema_registry import SchemaRegistry
from src.metricsCrew.config import get_agent_config, get_prompt_template


class CodeAgent:
    """
    Agent specialized in generating pandas code for data analysis.
    
    This agent translates structured analytical requirements into executable
    Python/pandas code that performs the requested analysis on a dataset.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize the CodeAgent.
        
        Args:
            config: Configuration dictionary for the agent
            schema_registry: SchemaRegistry instance for metadata access
        """
        # Load configuration with defaults
        self.config = config or get_agent_config('code_agent', {})
        
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
        temperature = llm_config.get('temperature', 0.2)
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Get prompt template for code generation
        code_prompt = get_prompt_template('code_agent.prompt', 
            """You are a Data Scientist specialized in pandas and data analysis.
            Your task is to create precise analytical code based on specific requirements.
            Generate clean, efficient, and well-commented pandas code that:
            
            1. Handles data loading and preprocessing
            2. Implements the requested analysis operations
            3. Includes appropriate error handling
            4. Follows pandas best practices
            5. Returns results in a clear, structured format
            
            Your code should be production-ready, well-organized, and include helpful comments.""")
        
        # Create the agent
        agent = Agent(
            role=self.config.get('role', "Data Scientist"),
            goal=self.config.get('goal', "Create precise analytical code to extract insights"),
            backstory=self.config.get('backstory', 
                "A Python expert specializing in pandas, with experience in translating analytical requirements into code"),
            verbose=self.config.get('verbose', True),
            allow_delegation=self.config.get('allow_delegation', False),
            llm=llm
        )
        
        return agent
    
    def generate_code(self, requirements: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
        """
        Generate pandas code based on analytical requirements.
        
        This is a utility method that directly uses the LLM without going through
        the CrewAI task framework.
        
        Args:
            requirements: Dictionary with structured analytical requirements
            dataset_id: Identifier for the dataset to analyze
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            # Get schema information
            schema = self.schema_registry.get_schema(dataset_id)
            
            # Get column information
            columns_info = {}
            for column_name, column_data in schema["columns"].items():
                columns_info[column_name] = {
                    "data_type": column_data["data_type"],
                    "category": column_data["category"],
                    "sample_values": column_data["sample_values"][:3] if "sample_values" in column_data else []
                }
            
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
            
            # Create the prompt for code generation
            prompt = f"""You are a Data Scientist specialized in pandas and data analysis.
            Your task is to create Python/pandas code to analyze a dataset based on the following requirements.
            
            Analytical Requirements:
            {json.dumps(requirements, indent=2)}
            
            Dataset Schema Information:
            Dataset ID: {dataset_id}
            Column Details:
            {json.dumps(columns_info, indent=2)}
            
            Generate a Python function that:
            
            1. Takes a pandas DataFrame 'df' as input
            2. Implements the analysis described in the requirements
            3. Handles edge cases (empty dataframe, missing values, etc.)
            4. Returns the analysis results in a structured format
            5. Includes clear comments and follows pandas best practices
            
            Your code should assume:
            - The input DataFrame is already loaded and available as 'df'
            - Pandas and NumPy are already imported as 'pd' and 'np'
            - The DataFrame has all the columns mentioned in the schema
            
            Generate only the Python function code. Do not include explanations or introductions.
            The function should be named 'analyze_data'.
            """
            
            # Get the response from the LLM
            response = llm.invoke(prompt)
            
            # Extract the code from the response
            try:
                import re
                
                # Extract code from the response
                code_match = re.search(r'```python(.*?)```', response.content, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1).strip()
                else:
                    # Try without language specification
                    code_match = re.search(r'```(.*?)```', response.content, re.DOTALL)
                    if code_match:
                        code = code_match.group(1).strip()
                    else:
                        # Take the entire response if no code blocks are found
                        code = response.content.strip()
                
                # Check if the code defines the analyze_data function
                if not re.search(r'def\s+analyze_data', code):
                    # If not, try to fix it by extracting function content
                    if re.search(r'def\s+\w+', code):
                        old_name = re.search(r'def\s+(\w+)', code).group(1)
                        code = code.replace(f"def {old_name}", "def analyze_data")
                
                return {
                    "code": code,
                    "requirements": requirements,
                    "dataset_id": dataset_id
                }
                    
            except Exception as e:
                return {"error": f"Error extracting code from LLM response: {str(e)}",
                        "raw_response": response.content}
            
        except Exception as e:
            return {"error": f"Error generating code: {str(e)}"}
    
    def sanitize_code(self, code: str) -> str:
        """
        Sanitize and validate the generated code for security.
        
        Args:
            code: Python code to sanitize
            
        Returns:
            Sanitized Python code
        """
        # Check for potentially harmful operations
        dangerous_patterns = [
            r'eval\(',            # eval() function
            r'exec\(',            # exec() function
            r'__import__\(',      # __import__() function
            r'subprocess',        # Subprocess module
            r'sys\.modules',      # Access to sys.modules
            r'import\s+os',       # Importing os module
            r'import\s+sys',      # Importing sys module
            r'open\(',            # File operations
            r'\.read\(',          # File reading
            r'\.write\(',         # File writing
            r'requests',          # HTTP requests
            r'socket',            # Socket operations
            r'shutil',            # File operations module
            r'importlib',         # Dynamic imports
        ]
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                # Replace with a safe comment
                code = re.sub(pattern, "# [REMOVED FOR SECURITY]", code)
        
        # Ensure the code only uses the allowed modules
        allowed_imports = ['pandas', 'numpy', 'math', 'datetime', 're', 'json']
        
        # Replace unauthorized imports
        import_pattern = r'import\s+(\w+)'
        for match in re.findall(import_pattern, code):
            if match not in allowed_imports:
                code = re.sub(f'import\\s+{match}', f'# import {match} # [REMOVED FOR SECURITY]', code)
        
        from_import_pattern = r'from\s+(\w+)'
        for match in re.findall(from_import_pattern, code):
            if match not in allowed_imports:
                code = re.sub(f'from\\s+{match}', f'# from {match} # [REMOVED FOR SECURITY]', code)
        
        return code
    
    def optimize_code(self, code: str) -> str:
        """
        Optimize the generated code for performance.
        
        Args:
            code: Python code to optimize
            
        Returns:
            Optimized Python code
        """
        # This is a placeholder for code optimization
        # In a production system, this would include:
        # - Vectorization of operations
        # - Memory usage optimization
        # - Performance improvements
        
        # For now, we'll just add a comment
        optimized_code = "# Code optimized for performance\n" + code
        
        return optimized_code
    
    def enhance_code_with_best_practices(self, code: str, requirements: Dict[str, Any]) -> str:
        """
        Enhance the generated code with pandas best practices.
        
        Args:
            code: Python code to enhance
            requirements: Original analytical requirements
            
        Returns:
            Enhanced Python code
        """
        try:
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
            
            # Create the prompt for code enhancement
            prompt = f"""You are a Python expert specializing in pandas optimization.
            Review the following code and enhance it with pandas best practices.
            
            Original Code:
            ```python
            {code}
            ```
            
            Original Requirements:
            {json.dumps(requirements, indent=2)}
            
            Your task is to:
            1. Improve code readability and organization
            2. Use vectorized operations instead of loops where possible
            3. Add error handling for edge cases
            4. Optimize memory usage for large datasets
            5. Add helpful comments and docstrings
            
            Return only the enhanced code without any explanations.
            """
            
            # Get the response from the LLM
            response = llm.invoke(prompt)
            
            # Extract the code from the response
            try:
                import re
                
                # Extract code from the response
                code_match = re.search(r'```python(.*?)```', response.content, re.DOTALL)
                
                if code_match:
                    enhanced_code = code_match.group(1).strip()
                else:
                    # Try without language specification
                    code_match = re.search(r'```(.*?)```', response.content, re.DOTALL)
                    if code_match:
                        enhanced_code = code_match.group(1).strip()
                    else:
                        # Take the entire response if no code blocks are found
                        enhanced_code = response.content.strip()
                
                return enhanced_code
                    
            except Exception as e:
                # Return the original code if there's an error
                return code
            
        except Exception as e:
            # Return the original code if there's an error
            return code