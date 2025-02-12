# main.py
from crewai import Crew, Task
from agents import TranslatorAgent, ConversionAgent, FinancialAgent, VisualizationAgent, SlackAgent
import os
import json
import time
from dotenv import load_dotenv
from crewai.agents.parser import AgentAction
from crewai.agents.crew_agent_executor import ToolResult


print(f"Current working directory: {os.getcwd()}")
load_dotenv()

output_directory = os.getcwd()
json_file_path = "agent_result.json"

def get_directory_snapshot(directory):
    snapshot = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                mtime = os.stat(full_path).st_mtime
            except Exception:
                mtime = None
            snapshot[full_path] = mtime
    return snapshot

def get_new_files(before_snapshot, after_snapshot):
    new_files = []
    for file_path in after_snapshot:
        if file_path not in before_snapshot:
            new_files.append(file_path)
        else:
            if before_snapshot[file_path] != after_snapshot[file_path]:
                new_files.append(file_path)
    return new_files

current_task_thought_count = 0
current_task_tool_usage_count = 0
current_task_start_time = None
current_task_file_snapshot = None

def my_step_callback(step_output):
    global current_task_thought_count, current_task_tool_usage_count
    print(f"\n--- Step Output Debugging ---\n{step_output}\n--------------------------\n")
    if isinstance(step_output, AgentAction):
        current_task_thought_count += 1
        print(f"LLM Call #{current_task_thought_count}: {step_output}")
    elif isinstance(step_output, ToolResult):
        current_task_tool_usage_count += 1
        print(f"Tool Use #{current_task_tool_usage_count}: {step_output}")

def reset_task_globals():
    global current_task_thought_count, current_task_tool_usage_count
    global current_task_start_time, current_task_file_snapshot
    current_task_thought_count = 0
    current_task_tool_usage_count = 0
    current_task_start_time = None
    current_task_file_snapshot = None

class MultiAgentCrew:
    def __init__(self, overall_tasks):
        self.overall_tasks = overall_tasks

    def run(self):
        translator_agent = TranslatorAgent().get_agent()
        conversion_agent = ConversionAgent().get_agent()
        financial_agent = FinancialAgent().get_agent()
        visualization_agent = VisualizationAgent().get_agent()
        slack_agent = SlackAgent().get_agent()

        overall_results = []
        
        for idx, task in enumerate(self.overall_tasks, start=1):
            print("\n==============================")
            print(f"Start {idx}")
            print("==============================\n")

            sub_tasks = []
            if "translation" in task:
                sub_tasks.append(Task(
                    description=task["translation"],
                    agent=translator_agent,
                    expected_output="Translation completed successfully."
                ))
            if "conversion" in task:
                sub_tasks.append(Task(
                    description=task["conversion"],
                    agent=conversion_agent,
                    expected_output="Image conversion completed successfully."
                ))
            if "financial" in task:
                sub_tasks.append(Task(
                    description=task["financial"],
                    agent=financial_agent,
                    expected_output="Financial analysis completed successfully."
                ))
            if "visualization" in task:
                sub_tasks.append(Task(
                    description=task["visualization"],
                    agent=visualization_agent,
                    expected_output="Chart generation completed successfully."
                ))
            if "slack" in task:
                sub_tasks.append(Task(
                    description=task["slack"],
                    agent=slack_agent,
                    expected_output="Slack summary sent successfully."
                ))

            reset_task_globals()
            global current_task_start_time, current_task_file_snapshot
            current_task_start_time = time.time()
            current_task_file_snapshot = get_directory_snapshot(output_directory)

            crew = Crew(
                agents=[translator_agent, conversion_agent, financial_agent, visualization_agent, slack_agent],
                tasks=sub_tasks,
                verbose=True,
                task_callback=lambda x: None,  
                step_callback=my_step_callback
            )
            result = crew.kickoff()

            execution_time = time.time() - current_task_start_time
            after_snapshot = get_directory_snapshot(output_directory)
            new_files = get_new_files(current_task_file_snapshot, after_snapshot)

            stats = {
                "thought_count": current_task_thought_count,
                "tool_usage_count": current_task_tool_usage_count,
                "execution_time_seconds": execution_time,
                "files_produced": new_files
            }

            overall_task_name = f"Overall Task {idx}"
            print(f"\{overall_task_name} success.")
            print(f"The Stats: {stats}\n")

            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
                    try:
                        existing_stats = json.load(f)
                    except json.JSONDecodeError:
                        existing_stats = {}
            else:
                existing_stats = {}

            existing_stats[overall_task_name] = stats
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(existing_stats, f, indent=4, ensure_ascii=False)

            overall_results.append(result)

        return overall_results

if __name__ == "__main__":
    overall_tasks = [
        {
            "translation": "Translate the content of the file `data/QA.txt` into Japanese. "
                            "Then, read the translated question and answer . "
                            "Finally, save the answers in the `translate` folder in a file named `answers.txt`.",
            "slack": "Send image conversion summary to Slack."

        },
        {
            "conversion": "Convert all images in the 'data/screenshots' folder from JPEG to PNG and save them in the 'my_image' folder.",
            "slack": "Send image conversion summary to Slack."
        },
        {
            "financial": "Read the documents from the 'data/financial_docs' folder, generate the financial report, and save it in the 'report_2025' folder as report.txt and report.json.",
            "visualization": "Generate charts based on the report_2025/report.json and save them in the 'report_2025' folder.",
            "slack": "Send financial analysis and chart generation summary to Slack."
        },
    ]

    crew = MultiAgentCrew(overall_tasks)
    results = crew.run()

    print("\nFinal Task Results:")
    for res in results:
        print(f"- {res}")
