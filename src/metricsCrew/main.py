"""
Main Application Module

This module serves as the entry point for the Business Data Analysis system,
providing a command-line interface for interacting with the system.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from src.metricsCrew.core.crew_manager import CrewManager
from src.metricsCrew.config import get_config, get_data_source_config


def setup_environment():
    """Set up the environment for the application."""
    dotenv_path = "/Users/home/dev/crewAI/.env"
    load_dotenv(dotenv_path)
    # Check for required environment variables
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set")
        print("Please set it with: export GOOGLE_API_KEY=your_api_key")
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Business Data Analysis System"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a dataset")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--dataset", "-d", help="Dataset ID to query")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dataset")
    analyze_parser.add_argument("path", help="Path to the dataset file")
    analyze_parser.add_argument("--id", help="Identifier for the dataset")
    analyze_parser.add_argument("--description", help="Description of the dataset")
    
    # List command
    subparsers.add_parser("list", help="List available datasets")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive query mode")
    
    return parser.parse_args()


def process_query(crew_manager: CrewManager, query: str, dataset_id: Optional[str] = None):
    """
    Process a query using the CrewManager.
    
    Args:
        crew_manager: CrewManager instance
        query: Query text
        dataset_id: Optional dataset ID to query
    """
    print(f"\nAnalyzing: '{query}'")
    print("This may take a minute or two...\n")
    
    # Execute the query
    result = crew_manager.analyze_query(query, dataset_id)
    
    # Check for errors
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print the result
    print("\n===== ANALYSIS RESULT =====\n")
    print(result["result"])
    print("\n===========================\n")


def process_analyze(crew_manager: CrewManager, path: str, dataset_id: Optional[str] = None,
                  description: Optional[str] = None):
    """
    Process a dataset analysis using the CrewManager.
    
    Args:
        crew_manager: CrewManager instance
        path: Path to the dataset file
        dataset_id: Optional identifier for the dataset
        description: Optional description of the dataset
    """
    print(f"\nAnalyzing dataset: '{path}'")
    print("This may take a minute...\n")
    
    # Execute the analysis
    result = crew_manager.analyze_dataset(path, dataset_id, description)
    
    # Check for errors
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print the result
    print("\n===== DATASET ANALYSIS =====\n")
    print(json.dumps(result, indent=2))
    print("\n============================\n")


def list_datasets(crew_manager: CrewManager):
    """
    List available datasets using the CrewManager.
    
    Args:
        crew_manager: CrewManager instance
    """
    datasets = crew_manager.get_available_datasets()
    
    if not datasets:
        print("No datasets available")
        return
    
    print("\n===== AVAILABLE DATASETS =====\n")
    
    for dataset_id, dataset_info in datasets.items():
        print(f"ID: {dataset_id}")
        print(f"Path: {dataset_info.get('path', 'N/A')}")
        print(f"Description: {dataset_info.get('description', 'N/A')}")
        print()
    
    print("=============================\n")


def interactive_mode(crew_manager: CrewManager):
    """
    Start interactive query mode.
    
    Args:
        crew_manager: CrewManager instance
    """
    print("\n===== Interactive Query Mode =====")
    print("Type 'exit' to quit, 'list' to show available datasets\n")
    
    while True:
        query = input("Query: ")
        
        if query.lower() == "exit":
            break
        elif query.lower() == "list":
            list_datasets(crew_manager)
            continue
        
        process_query(crew_manager, query)


def main():
    """Main application entry point."""
    # Set up the environment
    setup_environment()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create the crew manager
    crew_manager = CrewManager()
    
    # Process the command
    if args.command == "query":
        process_query(crew_manager, args.text, args.dataset)
    elif args.command == "analyze":
        process_analyze(crew_manager, args.path, args.id, args.description)
    elif args.command == "list":
        list_datasets(crew_manager)
    elif args.command == "interactive":
        interactive_mode(crew_manager)
    else:
        # Default to interactive mode if no command specified
        interactive_mode(crew_manager)


if __name__ == "__main__":
    main()