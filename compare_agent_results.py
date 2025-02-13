import pandas as pd
import json
import os

def load_data(files):
    structured_data = {}

    for agent_name, file_path in files.items():
        with open(file_path, "r") as f:
            content = json.load(f)

        for index, (task_name, details) in enumerate(content.items(), start=1):
            task_label = f"Task {index}"

            if task_label not in structured_data:
                structured_data[task_label] = {}

            thought_count = details.get("thought_count") or details.get("thoughts_count", 0)
            tool_usage_count = details.get("tool_usage_count") or details.get("tools_count", 0)
            execution_time = details.get("execution_time_seconds") or details.get("processing_time", 0)

            structured_data[task_label][f"{agent_name} - Thought Count"] = thought_count
            structured_data[task_label][f"{agent_name} - Tool Usage Count"] = tool_usage_count
            structured_data[task_label][f"{agent_name} - Execution Time (Seconds)"] = execution_time

    return structured_data

def save_csvs(structured_data, output_dir, files):
    df_agents_columns = pd.DataFrame.from_dict(structured_data, orient="index")


    columns_ordered = []
    for agent_name in files.keys():
        columns_ordered.append(f"{agent_name} - Thought Count")
        columns_ordered.append(f"{agent_name} - Tool Usage Count")
        columns_ordered.append(f"{agent_name} - Execution Time (Seconds)")

    df_agents_columns = df_agents_columns[columns_ordered]

    csv_path_1 = os.path.join(output_dir, "agents_comparison_by_columns.csv")
    df_agents_columns.to_csv(csv_path_1, index_label="Task Description")

    df_thoughts = df_agents_columns.filter(like="Thought Count")
    df_tools = df_agents_columns.filter(like="Tool Usage Count")
    df_execution = df_agents_columns.filter(like="Execution Time")

    df_thoughts.to_csv(os.path.join(output_dir, "agents_comparison_thoughts.csv"))
    df_tools.to_csv(os.path.join(output_dir, "agents_comparison_tools.csv"))
    df_execution.to_csv(os.path.join(output_dir, "agents_comparison_execution.csv"))

def main():
    files = {
        "CrewAI Multi-Agent": "crewai_multi_agent_agents_results.json",  # Add the file URL here
        "CrewAI Single-Agent": "crewai_single_agent_agents_results.json",  # Add the file URL here
        "OpenAI Swarm": "swarm_agent_results.json",  # Add the file URL here
        "XpressAI": "xpressai_agent_results.json"  # Add the file URL here
}


    output_dir = "csv_reports"
    os.makedirs(output_dir, exist_ok=True)

    structured_data = load_data(files)
    save_csvs(structured_data, output_dir, files)

    print("CSV files generated successfully in", output_dir)

if __name__ == "__main__":
    main()
