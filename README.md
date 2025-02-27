# Data Query AI Agents

## Overview

This project demonstrates an AI-powered natural language interface for querying data from multiple sources without requiring technical knowledge of the underlying data structures. Using a crew of specialized AI agents, the system can interpret natural language queries, retrieve relevant information from various data sources, perform analysis, and present results in a user-friendly format.

## 🚀 Features

- Natural language query interface for data retrieval
- Multiple agent architecture for specialized tasks
- Filesystem-based data sources (CSV, JSON, logs, etc.) in POC phase
- Extensible framework for adding database connectors (Splunk, MongoDB, Oracle)
- Automated data analysis and visualization capabilities
- Contextual response generation

## 🏗️ Architecture

The system uses a crew of specialized AI agents working together:

1. **Query Understanding Agent** - Interprets natural language requests
2. **Data Source Router Agent** - Determines which data sources to query
3. **Data Retrieval Agent** - Extracts information from identified sources
4. **Analysis Agent** - Processes and analyzes retrieved data
5. **Response Generation Agent** - Formats results with visualizations

![Agent Architecture](docs/agent_architecture.png)

## 📋 Requirements

- Python 3.9+
- CrewAI
- LangChain
- Pandas
- NumPy
- Matplotlib/Plotly
- Vector database (Chroma)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-query-ai.git
cd data-query-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file with your API keys
```

## 🔍 Usage

### Basic Query

```python
from data_query import DataQueryCrew

# Initialize crew
crew = DataQueryCrew()

# Run a query
result = crew.query("Show me sales performance by region for Q4 2023")

# Display results
print(result.summary)
result.display_visualization()
```

### Sample Queries

```
"What were our top-performing products last month?"
"Compare customer satisfaction scores across different regions"
"Show me system performance metrics from last weekend's outage"
"Generate a report on market trends for the healthcare sector"
```

## ⚙️ Configuration

### Data Sources

Configure data sources in `config/data_sources.yaml`:

```yaml
sources:
  - name: sales_data
    type: csv
    path: data/sales/*.csv
    description: "Sales transaction data including region, product, amount"
    
  - name: customer_feedback
    type: json
    path: data/feedback/*.json
    description: "Customer satisfaction surveys and feedback"
    
  - name: system_logs
    type: log
    path: data/logs/*.log
    description: "System performance logs with timestamp, service, and metrics"
```

### Agent Configuration

Customize agent behavior in `config/agents.yaml`.

## 🛠️ Development

### Adding New Data Sources

1. Implement a connector class in `src/connectors/`
2. Register the connector in `src/registry.py`
3. Add configuration in `config/data_sources.yaml`

### Creating Custom Tools

1. Define tool class in `src/tools/`
2. Implement required methods
3. Register with appropriate agent

## 🔮 Future Roadmap

- [ ] Add connectors for Splunk, MongoDB, and Oracle
- [ ] Implement conversational context for follow-up queries
- [ ] Add support for real-time data streams
- [ ] Develop security and access control layer
- [ ] Create web UI for non-technical users

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📞 Contact

For questions or support, please open an issue or contact the project maintainers at [your-email@example.com](mailto:your-email@example.com).