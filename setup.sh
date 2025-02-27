#!/bin/bash

# Create main directory structure
mkdir -p src/metricsCrew/agents
mkdir -p src/metricsCrew/config
mkdir -p src/metricsCrew/core
mkdir -p src/metricsCrew/tasks
mkdir -p src/metricsCrew/tools
mkdir -p src/metricsCrew/utils
mkdir -p tests/dataset

# Create Python __init__.py files
touch src/metricsCrew/__init__.py
touch src/metricsCrew/agents/__init__.py
touch src/metricsCrew/config/__init__.py
touch src/metricsCrew/core/__init__.py
touch src/metricsCrew/tasks/__init__.py
touch src/metricsCrew/tools/__init__.py
touch src/metricsCrew/utils/__init__.py
touch tests/__init__.py

# Create Python files
touch src/metricsCrew/main.py
touch src/metricsCrew/agents/manager_agent.py
touch src/metricsCrew/agents/schema_agent.py
touch src/metricsCrew/agents/query_agent.py
touch src/metricsCrew/agents/code_agent.py
touch src/metricsCrew/agents/result_agent.py
touch src/metricsCrew/core/data_manager.py
touch src/metricsCrew/core/schema_registry.py
touch src/metricsCrew/core/context_manager.py
touch src/metricsCrew/core/crew_manager.py
touch src/metricsCrew/tasks/task_definitions.py
touch src/metricsCrew/tools/schema_tool.py
touch src/metricsCrew/tools/code_tool.py
touch src/metricsCrew/tools/result_tool.py
touch src/metricsCrew/tools/visualization_tool.py
touch src/metricsCrew/utils/gemini_client.py
touch src/metricsCrew/utils/validators.py

# Create test files
touch tests/test_schema_analysis.py
touch tests/test_query_interpretation.py
touch tests/test_code_generation.py
touch tests/test_result_interpretation.py
touch tests/test_end_to_end.py

# Create YAML files
touch src/metricsCrew/config/config.yaml
touch src/metricsCrew/config/agents.yaml
touch src/metricsCrew/config/tasks.yaml
touch src/metricsCrew/config/tools.yaml
touch src/metricsCrew/config/prompts.yaml
touch src/metricsCrew/config/data_sources.yaml

echo "Project structure created successfully!"
