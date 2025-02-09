import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import dotenv

os.environ['OPENAI_API_KEY'] = 'your open API_key'
# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Print to verify the API key is set (don't do this in production!)
print(f"API key is {'set' if OPENAI_API_KEY else 'not set'}")

# Initialize the LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

# Create a simple agent
researcher = Agent(  
    role='Researcher',
    goal='Research and analyze topics thoroughly',
    backstory='You are an expert researcher with extensive experience',
    llm=llm,
    verbose=True
)

# Create a simple task
task = Task(
    description="Research the benefits of exercise",
    expected_output="A brief summary of the main benefits of regular exercise",
    agent=researcher
)

# Create the crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True
)

# Run the crew
result = crew.kickoff()
print("\nResult:", result)