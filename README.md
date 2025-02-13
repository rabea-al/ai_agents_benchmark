# AI Agents Benchmarking

This repository provides a benchmarking framework for evaluating different AI agent architectures, including **CrewAI, Swarm, and XpressAI**. Each agent specializes in various tasks such as **translation, image conversion, financial analysis, and report generation**.

##  Repository Structure
The repository is structured as follows:

- **`/xpressai/`** → Agent using XpressAI platform
- **`/crewai_multi_agent/`** → Multi-agent system based on CrewAI
- **`/crewai_single_agent/`** → Single-agent system based on CrewAI
- **`/swarm/`** → Single-agent system based on Swarm framework
- **`/compare_agent_results.py/`** → Script to compare performance results


Each folder contains its own `README.md` with specific details about that agent.



##  Comparing Agent Performance
The `compare_agent_results.py` script aggregates and compares results from all agents. It reads the output JSON files for each agent and compiles structured **CSV reports** that allow easy analysis. To run it:

```bash
python compare_agent_results.py
```

### **Generated CSV Reports**
- `agents_comparison_by_columns.csv` → Full comparison table of all metrics
- `agents_comparison_thoughts.csv` → Thought count comparison
- `agents_comparison_tools.csv` → Tool usage comparison
- `agents_comparison_execution.csv` → Execution time comparison

To ensure correct operation, update `compare_agent_results.py` with the correct paths to each agent’s `agent_results.json` file before running the script.


