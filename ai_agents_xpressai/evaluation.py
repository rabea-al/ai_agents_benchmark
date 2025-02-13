import os
import json
import requests
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API setup
GPT_API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("OPENAI_API_KEY")

def evaluate_task(task):
    task_type = task["task_type"]
    input_files = task["input_files"]
    output_files = task["output_files"]
    results = {}

    if task_type == "image_conversion":
        for output_file in output_files.values():
            results[output_file] = os.path.exists(output_file) and output_file.endswith(".png")

    elif task_type == "translation":
        with open(list(input_files.values())[0], "r") as f:
            input_text = f.read()
        with open(list(output_files.values())[0], "r") as f:
            translated_text = f.read()
        prompt = (
            f"Input text:\n{input_text}\n\n"
            f"Translated text:\n{translated_text}\n\n"
            f"Evaluate the translation quality from 1 to 10, considering accuracy and fluency. "
            f"Provide concise feedback (max 30 words)."
        )
        response = send_to_gpt(prompt)
        results["evaluation"] = response.get("rating", "Error: No rating received")
        results["feedback"] = response.get("feedback", "No feedback received")

    elif task_type == "chart_generation":
        for output_file in output_files.values():
            results[output_file] = os.path.exists(output_file)

    elif task_type == "report_generation":
        with open(list(input_files.values())[0], "r") as f:
            input_data = f.read()
        with open(list(output_files.values())[0], "r") as f:
            report_content = f.read()
        prompt = (
            f"Input financial data:\n{input_data}\n\n"
            f"Generated report:\n{report_content}\n\n"
            f"Evaluate the report quality from 1 to 10, considering accuracy and insights. "
            f"Provide concise feedback (max 30 words)."
        )
        response = send_to_gpt(prompt)
        results["evaluation"] = response.get("rating", "Error: No rating received")
        results["feedback"] = response.get("feedback", "No feedback received")

    return results

def send_to_gpt(prompt):
    if not API_KEY:
        print("Error: OPENAI_API_KEY is missing in the .env file")
        return {"rating": None, "feedback": "Missing API Key"}

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        print(f"Sending prompt to GPT:\n{prompt}\n")
        response = requests.post(GPT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        gpt_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_gpt_response(gpt_content)
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with GPT: {e}")
        return {"rating": None, "feedback": str(e)}

def parse_gpt_response(response_content):
    rating_match = re.search(r"\b([1-9]|10)\b", response_content)
    rating = int(rating_match.group(0)) if rating_match else None
    feedback = " ".join(response_content.split()[:30])
    return {"rating": rating, "feedback": feedback}

def evaluate_all_tasks(task_file):
    with open(task_file, "r") as file:
        tasks = json.load(file)["tasks"]

    results = {}
    for i, task in enumerate(tasks, start=1):
        print(f"Evaluating Task {i}: {task['task_description']}")
        task_results = evaluate_task(task)
        results[f"task_{i}"] = task_results

    with open("evaluation_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("All tasks evaluated. Results saved in evaluation_results.json")

evaluate_all_tasks("task.json")
