from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List
import os

os.environ['OPENAI_API_KEY'] = 'your key'
class ResearchCrew:
    def __init__(self, openai_api_key: str):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Create the researcher agent
        self.researcher = Agent(
            role='Research Analyst',
            goal='Conduct thorough research on given topics and gather key information',
            backstory="""You are an expert research analyst with years of experience in 
            gathering and analyzing information. You excel at breaking down complex topics 
            into key components and finding relevant insights.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        # Create the writer agent
        self.writer = Agent(
            role='Content Writer',
            goal='Transform research findings into clear, well-structured reports',
            backstory="""You are a skilled content writer specialized in creating 
            comprehensive yet accessible reports. You excel at organizing information 
            logically and presenting it in an engaging way.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

    def create_research_tasks(self, topic: str) -> List[Task]:
        """Create research and writing tasks for the given topic."""
        
        # Research task
        research_task = Task(
            description=f"""Research the following topic in detail: {topic}
            
            1. Identify key concepts and components
            2. Find relevant facts and statistics
            3. Note any important debates or controversies
            4. Look for recent developments
            
            Compile your findings in a structured format.""",
            expected_output="""A comprehensive research document containing:
            - Key concepts and definitions
            - Important statistics and data
            - Current debates and controversies
            - Recent developments and trends
            - Citations and sources""",
            agent=self.researcher
        )
        
        # Writing task
        writing_task = Task(
            description=f"""Using the research provided, create a comprehensive report on: {topic}
            
            The report should include:
            1. An executive summary
            2. Key findings and insights
            3. Supporting evidence and examples
            4. Conclusions and implications
            
            Make the report engaging and accessible while maintaining accuracy.""",
            expected_output="""A well-structured report containing:
            - Executive summary
            - Main findings and analysis
            - Supporting evidence
            - Conclusions and recommendations
            - Professional formatting""",
            agent=self.writer
        )
        
        return [research_task, writing_task]

    def run_research(self, topic: str) -> str:
        """Execute the research process for a given topic."""
        try:
            # Create the crew with verbose set to True instead of 2
            crew = Crew(
                agents=[self.researcher, self.writer],
                tasks=self.create_research_tasks(topic),
                verbose=True,  # Changed from 2 to True
                process=Process.sequential
            )
            
            # Start the crew's work
            result = crew.kickoff()
            
            return result
            
        except Exception as e:
            return f"Error during research process: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize the research crew
    
    # Example research topics
    topics = [
        "The impact of artificial intelligence on healthcare",
        "Sustainable energy solutions for urban environments"
    ]
    
    # Run research for each topic
    for topic in topics:
        print(f"\nResearching topic: {topic}")
        result = crew.run_research(topic)
        print(f"\nResults:\n{result}")