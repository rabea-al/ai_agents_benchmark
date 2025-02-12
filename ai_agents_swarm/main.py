import os
import json
import time
from dotenv import load_dotenv
from swarm import Swarm
from agents import multi_agent
from collections import Counter

load_dotenv()

swarm = Swarm()

def execute_task(task_description: str) -> dict:
   
    messages = [{"role": "user", "content": task_description}]
    response = swarm.run(agent=multi_agent, messages=messages)

    return {
        "messages": response.messages if hasattr(response, "messages") else str(response),
        "status": "completed"
    }

def analyze_results(task_stats_file="task_stats.json", output_file="agent_results.json"):
    
    with open(task_stats_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    agent_results = {}

    for idx, (task, details) in enumerate(data.items(), start=1):
        messages = details["result"]["messages"]
        execution_time = details["execution_time_seconds"]

        thought_count = sum(1 for msg in messages if msg["role"] == "assistant" and msg.get("tool_calls"))

        tool_usage_count = sum(len(msg["tool_calls"]) for msg in messages if msg.get("tool_calls"))

        files_produced = []
        for msg in messages:
            if msg["role"] == "tool" and msg.get("content"):
                content = msg["content"]
                if "saved to" in content or "File successfully saved" in content:
                    file_path = content.split("saved to")[-1].strip()
                    files_produced.append(file_path)

        agent_results[f"Overall Task {idx}"] = {
            "thought_count": thought_count,
            "tool_usage_count": tool_usage_count,
            "execution_time_seconds": execution_time,
            "files_produced": files_produced
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(agent_results, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis completed. Results saved in {output_file}")

def main():
    tasks = [
        "Translate the text inside the file data/QA.txt into Japanese, then answer the translated questions and save the answers in the translate folder in a file named answers.txt, and send summary to Slack.",
        "Convert all JPEG images in data/screenshots/ to PNG and save them in my_image, then send summary to Slack.",
        "Read the documents from the data/financial_docs folder, generate the financial report, and save the financial report in the report_2025 folder as report.txt and save it as report.json as well, After that, generate the charts and save them in report_2025 and send summary of result to Slack."
    ]
    
    overall_results = {}
    for task in tasks:
        print(f"\nExecuting task: {task}")
        start_time = time.time()
        result = execute_task(task)
        elapsed_time = time.time() - start_time
        print(f"Result: {result}")

        overall_results[task] = {
            "result": result,
            "execution_time_seconds": elapsed_time
        }

    with open("task_stats.json", "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=4, ensure_ascii=False)

    print("\nAll tasks completed. Now analyzing results...")

    analyze_results()

if __name__ == "__main__":
    main()
