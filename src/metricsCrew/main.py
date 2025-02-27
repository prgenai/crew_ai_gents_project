import sys
import json
from src.metricsCrew.agents.agent_manager import AgentManager
from src.metricsCrew.core.crew_manager import CrewManager
from src.metricsCrew.core.config import Config

def print_header():
    """Print application header"""
    print("\n" + "=" * 80)
    print(" " * 25 + "AI METRICS AGENT SYSTEM")
    print("=" * 80)

def print_help():
    """Print help information"""
    print("\nAvailable commands:")  
    print("  query [your question]  - Ask a question about application metrics")
    print("  complex [your question] - Ask a complex question (uses multiple agents)")
    print("  help                   - Show this help information")
    print("  exit                   - Exit the application")
    print("\nExample queries:")
    print("  query Show performance metrics for PaymentAPI")
    print("  query Analyze error patterns in the last 24 hours")
    print("  query Check CPU usage across all services")
    print("  query Display business metrics for the last month")
    print("  complex Why did we see a drop in revenue yesterday?")

def format_output(result):
    """Format the output in a readable way"""
    return json.dumps(result, indent=2)

def main():
    """Main application entry point"""
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize agent and crew managers
        agent_manager = AgentManager()
        crew_manager = CrewManager(agent_manager)
        
        print_header()
        print("\nAI Metrics Agent initialized and ready.")
        print_help()
        
        # Command loop
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                # Process commands
                if user_input.lower() == 'exit':
                    print("Exiting application...")
                    break
                elif user_input.lower() == 'help':
                    print_help()
                elif user_input.lower().startswith('query '):
                    query = user_input[6:].strip()
                    if not query:
                        print("Please provide a query. Example: query Show performance metrics")
                        continue
                    
                    print(f"\nProcessing query: {query}")
                    print("This may take a moment...\n")
                    
                    result = crew_manager.process_query(query)
                    print("\nRESULT:")
                    print("-" * 80)
                    print(result["result"])
                    print("-" * 80)
                elif user_input.lower().startswith('complex '):
                    query = user_input[8:].strip()
                    if not query:
                        print("Please provide a complex query.")
                        continue
                    
                    print(f"\nProcessing complex query: {query}")
                    print("This may take some time as multiple agents work on your request...\n")
                    
                    result = crew_manager.process_complex_query(query)
                    print("\nRESULT:")
                    print("-" * 80)
                    print(result["result"])
                    print("-" * 80)
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled. Type 'exit' to quit or continue with a new query.")
            except Exception as e:
                print(f"Error processing input: {str(e)}")
    
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        print("Please set up your .env file with required values.")
        return 1
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())