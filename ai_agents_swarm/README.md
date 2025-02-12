# ai_agents_Swarm_Single_Agent

The system utilizes a **Swarm-based single-agent architecture**, where one agent is responsible for handling multiple tasks such as translation, image conversion, financial analysis, and chart generation. The agent leverages **GPT-4o** for decision-making and parameter extraction, and it integrates external tools to execute tasks efficiently. The Swarm framework ensures seamless task coordination, real-time processing, and structured output storage with automated notifications.

---

## Prerequisites
Before running the agent, ensure that the following dependencies are installed and configured:

### 1. **Install Requirements**:
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables:
Ensure you have a `.env` file containing the following variables and replace `<your_value>` with your actual credentials:

```bash
SLACK_SERVER_URL=<your_server_url>
SLACK_CHANNEL_ID=<your_channel_id>
SLACK_BOT_TOKEN=<your_bot_token>
DEEPL_API_KEY=<your_deepl_api_key>
OPENAI_API_KEY=<your_openai_api_key>
```

---

## How to Run the Agent

After setting up the environment and installing dependencies, execute the `main.py` file:

```bash
python main.py
```

---
## Workflow

The system operates as follows:

1. **Agent Definition**  
   - A single Swarm-based agent (`multi_agent`) is responsible for executing various tasks.
   - The agent utilizes **GPT-4o** for decision-making and external tools for execution.

2. **Task Execution**  
   - Each task is passed as a natural language command.
   - The agent processes the request using Swarm and executes relevant tools accordingly.

3. **Processing & Monitoring**  
   - Execution details such as thought count, tool usage, and file creation are tracked.

4. **Saving Results**  
     The system processes tasks and generates the following outputs:

    - **Translation Output**  
        - `answers.txt`: Translated answers from `QA.txt` into Japanese.

    - **Image Conversion Output**  
        - Converted images saved in the specified directory.

    - **Financial Analysis Output**  
        - Financial reports generated and stored as `.txt` and `.json`.

    - **Visualization Output**  
        - Financial charts created based on analysis data.

    - **Execution Summary**  
        - `agent_results.json`: Contains execution metrics like tool usage count and files produced.
        - The agent generates a `task_stats.json` file that logs execution details for each task, including responses, tool calls, and output files. The system then processes this file to extract relevant metrics and compiles them into `agent_results.json`, which provides a structured summary of execution performance.

5. **Sending Results**
   - Summaries of completed tasks are automatically sent to Slack.

---
### Additional Resources

The agent operates using predefined resources:

- **financial_docs**: A folder containing financial documents for generating reports and charts.
- **screenshots**: A folder with sample images for conversion.
- **QA.txt**: A test file for translation tasks.

---

### Available Tools

The agent is equipped with the following tools to execute tasks:

1. **Translation Tool**: Translates text files into the target language.
2. **Image Conversion Tool**: Converts images between different formats.
3. **Financial Analysis Tool**: Reads financial documents and generates analysis reports.
4. **Chart Generation Tool**: Creates financial charts based on analysis data.
5. **File Writing Tool**: Saves generated outputs (translations, reports, etc.) to files.
6. **Slack Integration**: Sends summaries and results to a Slack channel.

---

## Evaluation

The system includes an evaluation module using **GPT-4o** to verify the accuracy of outputs. Running `evaluation.py` performs the following checks:

- **Image Conversion**: Ensures images were properly converted.
- **Chart Generation**: Confirms that charts were generated and stored correctly.
- **Translation Quality**: Rates translations from 1 to 10 and provides feedback.
- **Report Evaluation**: Compares generated reports with source data and assigns a score from 1 to 10.

Evaluation depends on `tasks.py` for input/output mapping.

