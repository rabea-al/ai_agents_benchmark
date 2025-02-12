from crewai import Crew, Task
from agents import TranslatorAgents
import os
import json
import time
from dotenv import load_dotenv
from crewai.agents.parser import AgentAction
from crewai.agents.crew_agent_executor import ToolResult

load_dotenv()

output_directory = os.getcwd()

def get_directory_snapshot(directory):
    """
   
    """
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
    """
    """
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

json_file_path = "agent_result.json"

def my_step_callback(step_output):
    global current_task_thought_count, current_task_tool_usage_count
    print(f"\n--- Step Output Debugging ---\n{step_output}\n--------------------------\n")
    if isinstance(step_output, AgentAction):
        current_task_thought_count += 1
        print(f"LLM Call #{current_task_thought_count}: {step_output}")
    elif isinstance(step_output, ToolResult):
        current_task_tool_usage_count += 1
        print(f"Tool Use #{current_task_tool_usage_count}: {step_output}")

def my_task_callback(task_output):
    global current_task_thought_count, current_task_tool_usage_count
    global current_task_start_time, current_task_file_snapshot

    execution_time = time.time() - current_task_start_time if current_task_start_time is not None else 0

    task_name = task_output.description
    print(f"\nTask Completed: {task_name}")
    print(f"Output: {task_output.raw}\n")

    time.sleep(1)

    after_snapshot = get_directory_snapshot(output_directory)
    new_files = get_new_files(current_task_file_snapshot, after_snapshot) if current_task_file_snapshot is not None else []

    stats = {
        "thought_count": current_task_thought_count,
        "tool_usage_count": current_task_tool_usage_count,
        "execution_time_seconds": execution_time,
        "files_produced": new_files
    }

    print(f"New files created: {new_files}")

    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            try:
                existing_stats = json.load(f)
            except json.JSONDecodeError:
                existing_stats = {}
    else:
        existing_stats = {}
    existing_stats[task_name] = stats
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_stats, f, indent=4, ensure_ascii=False)

    reset_task_globals()

def reset_task_globals():
    global current_task_thought_count, current_task_tool_usage_count
    global current_task_start_time, current_task_file_snapshot
    current_task_thought_count = 0
    current_task_tool_usage_count = 0
    current_task_start_time = None
    current_task_file_snapshot = None

class TranslatorCrew:
    def __init__(self, tasks):
        self.tasks = tasks

    def run(self):
        agents = TranslatorAgents()
        translator_agent = agents.translator_agent()
        overall_results = []

        for task in self.tasks:
            print("\n==============================")
            print(f"Start {task['description']}")
            print("==============================\n")
            reset_task_globals()
            global current_task_start_time, current_task_file_snapshot
            current_task_start_time = time.time()
            current_task_file_snapshot = get_directory_snapshot(output_directory)

            crew_task = Task(
                description=task["description"],
                agent=translator_agent,
                expected_output="Task successfully completed."
            )
            crew = Crew(
                agents=[translator_agent],
                tasks=[crew_task],
                verbose=True,
                task_callback=my_task_callback,
                step_callback=my_step_callback
            )
            result = crew.kickoff()
            overall_results.append(result)
        return overall_results

if __name__ == "__main__":
    tasks = [
        {
            "description": "Translate the text inside the file data/QA.txt into Japanese, then answer the translated questions and save the answers in the translate folder in a file named answers.txt, and send summary to Slack."
        },
        {
            "description": "Convert all images in the data/screenshots folder from JPEG to PNG and save them in the my_image folder, and send summary of result to Slack."
        },
        {
            "description": "Read the documents from the data/financial_docs folder, generate the financial report, and save the financial report in the report_2025 folder as report.txt and save it as report.json as well. After that, generate the charts and save them in report_2025, and send summary of result to Slack."
        },
     ]

    translator_crew = TranslatorCrew(tasks)
    results = translator_crew.run()

    print("\nFinal Task Results:")
    for res in results:
        print(f"- {res}")
