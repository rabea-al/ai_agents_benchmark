import requests
import json

# Define the URL of your agent
AGENT_URL = "http://127.0.0.1:8080/chat/completions"

# Define the file path containing the prompts in JSON format
FILE_PATH = "tasks.json"

def send_prompt_to_agent(prompt):
    """Send a single prompt to the agent and return the response."""
    data = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(AGENT_URL, json=data)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the agent: {e}")
        return None

def main():
    """Read prompts from a JSON file and send them to the agent."""
    try:
        with open(FILE_PATH, "r") as file:
            prompts = json.load(file)  # Load JSON data
            if not isinstance(prompts, list):
                print("Invalid JSON format: Expected a list of prompts.")
                return

            for line_number, prompt_entry in enumerate(prompts, start=1):
                prompt = prompt_entry.get("prompt", "").strip()  # Read the 'prompt' key
                if not prompt:
                    print(f"Skipping empty prompt at entry {line_number}.")
                    continue

                print(f"Sending prompt {line_number}: {prompt}")
                response = send_prompt_to_agent(prompt)

                if response:
                    print(f"Response for prompt {line_number}: {response}\n")
                else:
                    print(f"No response for prompt {line_number}.\n")
    except FileNotFoundError:
        print(f"The file {FILE_PATH} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON file {FILE_PATH}. Ensure it is correctly formatted.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
