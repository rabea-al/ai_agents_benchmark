from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, secret

import abc
from collections import deque
from typing import NamedTuple

import json
import os
import requests
import random
import string
import copy
import matplotlib.pyplot as plt
import pandas as pd
from translate import Translator
from PIL import Image
import openai
from openai import OpenAI
from slack_sdk import WebClient

@xai_component
class AgentBenchmarkRun(Component):
    """Run the agent with the given conversation.

    ##### branches:
    - on_thought: Called whenever the agent uses a tool.

    ##### inPorts:
    - agent_name: The name of the agent to run.
    - conversation: The conversation to send to the agent.

    ##### outPorts:
    - out_conversation: The conversation with the agent's responses.
    - last_response: The last response of the agent.

    """
    on_thought: BaseComponent

    agent_name: InCompArg[str]
    conversation: InCompArg[any]

    out_conversation: OutArg[list]
    last_response: OutArg[str]

    def execute(self, ctx) -> None:
        agent = ctx['agent_' + self.agent_name.value]
        start_time = time.perf_counter()
        output_directory = "/data/home"
        before_files = set(os.listdir(output_directory))

        if agent['agent_provider'] == 'vertexai':
            model_name = agent['agent_model']
            toolbelt = agent['agent_toolbelt']
            system_prompt = agent['agent_system_prompt']

            # deep to avoid messing with the original system prompt.
            conversation = copy.deepcopy(self.conversation.value)

            if conversation[0]['role'] != 'system':
                conversation.insert(0, {'role': 'system', 'content': system_prompt.format(**make_tools_prompt(toolbelt))})
            else:
                conversation[0]['content'] =  system_prompt.format(**make_tools_prompt(toolbelt))

            thoughts = 0
            stress_level = 0.0 # Raise temperature if there are failures.

            while thoughts < agent['max_thoughts']:
                thoughts += 1

                if thoughts == agent['max_thoughts']:
                    conversation.append({"role": "system", "content": "Maximum tool usage reached.  Tools Unavailable"})


                inputs = conversation_to_vertexai(conversation)

                model = GenerativeModel(model_name)
                result = model.generate_content(
                    inputs,
                    generation_config={
                        "max_output_tokens": 2048,
                        "stop_sequences": [
                            "\n\nsystem:",
                            "\n\nuser:",
                            "\n\nassistant:"
                        ],
                        "temperature": stress_level + 0.5,
                        "top_p": 1
                    },
                    safety_settings=[],
                    stream=False,
                )

                if "assistant:" in result.text:
                    response = {"role": "assistant", "content": result.text.split("assistant:")[-1]}
                else:
                    response = {"role": "assistant", "content": result.text}

                conversation.append(response)

                if thoughts <= agent['max_thoughts'] and 'TOOL:' in response['content']:

                    next_action = self.on_thought
                    while next_action:
                        next_action = next_action.do(ctx)

                    lines = response['content'].split("\n")
                    for line in lines:
                        if line.startswith("TOOL:"):
                            command = line.split(":", 1)[1].strip()
                            try:
                                tool_name = command.split(" ", 1)[0].strip()
                                tool_args = command.split(" ", 1)[1].strip()
                            except Exception as e:
                                tool_name = command.strip()
                                tool_args = ""

                            if tool_name == 'recall':
                                memory = agent['agent_memory']
                                tool_result = str(memory.query(tool_args, 3))
                                print(f"recall got result: {tool_result}", flush=True)
                                conversation.append({"role": "system", "content": tool_result})
                            elif tool_name == 'remember':

                                memory = agent['agent_memory']
                                prompt_start = tool_args.find('"')
                                prompt_end = tool_args.find('"', prompt_start)
                                prompt = tool_args[prompt_start + 1:prompt_end].strip()
                                memo_start = tool_args.find('"', prompt_end)
                                memo = tool_args[memo_start + 1:len(tool_args - 1)].replace('\"', '"')

                                try:
                                    json_memo = json.loads(memo)
                                except Exception as e:
                                    # Invalid JSON, so just store as a string.
                                    json_memo = '"' + memo + '"'

                                memory.add('', prompt, json_memo)
                                print(f"Added {prompt}: {memo} to memory", flush=True)
                                conversation.append({"role": "system", "content": f"Memory {prompt} stored."})

                            else:
                                try:
                                    tool_result = toolbelt[tool_name](tool_args)
                                    print(f"tool {tool_name} got result:")
                                    print(tool_result)

                                    conversation.append({"role": "system", "content": tool_result})
                                except KeyError as e:
                                    print(f"tool {tool_name} not found.")
                                    conversation.append({"role": "system", "content": "ERROR: Tool not available: " + str(e)})
                                    stress_level = min(stress_level + 0.1, 1.5)
                                except Exception as e:
                                    print(f"tool {tool_name} got exception:")
                                    print(e)
                                    conversation.append({"role": "system", "content": "ERROR: " + str(e)})
                                    stress_level = min(stress_level + 0.1, 1.5)
                else:
                    # Allow only one tool per thought.
                    break
            self.out_conversation.value = conversation
            self.last_response.value = conversation[-1]['content']



        elif agent['agent_provider'] == 'openai':
            model_name = agent['agent_model']
            toolbelt = agent['agent_toolbelt']
            system_prompt = agent['agent_system_prompt']

            # deep to avoid messing with the original system prompt.
            conversation = copy.deepcopy(self.conversation.value)

            if conversation[0]['role'] != 'system':
                conversation.insert(0, {'role': 'system', 'content': system_prompt.format(**make_tools_prompt(toolbelt))})
            else:
                conversation[0]['content'] =  system_prompt.format(**make_tools_prompt(toolbelt))

            thoughts = 0
            stress_level = 0.0 # Raise temperature if there are failures.
            tool_usage_count = 0

            while thoughts <= agent['max_thoughts']:
                thoughts += 1

                if thoughts == agent['max_thoughts']:
                    conversation.append({"role": "system", "content": "Maximum tool usage reached.  Tools Unavailable"})

                result = openai.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    max_tokens=2000,
                    temperature=stress_level
                )

                response = result.choices[0].message

                conversation.append({"role": "assistant", "content": response.content})

                self.out_conversation.value = conversation
                self.last_response.value = conversation[-1]['content']

                if thoughts <= agent['max_thoughts'] and 'TOOL:' in response.content:
                    tool_usage_count += 1

                    next_action = self.on_thought
                    while next_action:
                        next_action = next_action.do(ctx)

                    lines = response.content.split("\n")
                    for line in lines:
                        if line.startswith("TOOL:"):
                            command = line.split(":", 1)[1].strip()
                            try:
                                tool_name = command.split(" ", 1)[0].strip()
                                tool_args = command.split(" ", 1)[1].strip()
                            except Exception as e:
                                tool_name = command.strip()
                                tool_args = ""

                            if tool_name == 'recall':
                                memory = agent['agent_memory']
                                tool_result = str(memory.query(tool_args, 3))
                                print(f"recall got result: {tool_result}", flush=True)
                                conversation.append({"role": "system", "content": tool_result})
                            elif tool_name == 'remember':
                                memory = agent['agent_memory']
                                prompt_start = tool_args.find('"')
                                prompt_end = tool_args.find('"', prompt_start)
                                prompt = tool_args[prompt_start + 1:prompt_end].strip()
                                memo_start = tool_args.find('"', prompt_end)
                                memo = tool_args[memo_start + 1:len(tool_args - 1)].replace('\"', '"')

                                try:
                                    json_memo = json.loads(memo)
                                except Exception as e:
                                    # Invalid JSON, so just store as a string.
                                    json_memo = '"' + memo + '"'

                                memory.add('', prompt, json_memo)
                                print(f"Added {prompt}: {memo} to memory", flush=True)
                                conversation.append({"role": "system", "content": f"Memory {prompt} stored."})

                            else:
                                try:
                                    tool_result = toolbelt[tool_name](tool_args)
                                    print(f"tool {tool_name} got result:")
                                    print(tool_result)

                                    conversation.append({"role": "system", "content": tool_result})
                                except KeyError as e:
                                    print(f"tool {tool_name} not found.")
                                    conversation.append({"role": "system", "content": "ERROR: Tool not available: " + str(e)})
                                    stress_level = min(stress_level + 0.1, 1.5)
                                except Exception as e:
                                    print(f"tool {tool_name} got exception:")
                                    print(e)
                                    conversation.append({"role": "system", "content": "ERROR: " + str(e)})
                                    stress_level = min(stress_level + 0.1, 1.5)
                else:
                    # Allow only one tool per thought.
                    break

            after_files = set(os.listdir(output_directory))

            new_entries = after_files - before_files
            new_files = []

            for entry in new_entries:
                entry_path = os.path.join(output_directory, entry)
                if os.path.isdir(entry_path):
                    for root, _, files in os.walk(entry_path):
                        for file in files:
                            new_files.append(os.path.join(root, file))
                else:
                    new_files.append(entry_path)

            print(f"New files created: {new_files}")

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.out_conversation.value = conversation
            self.last_response.value = conversation[-1]['content']
            print(f"Total thoughts: {thoughts}")
            print(f"Total tool usage in OpenAI: {tool_usage_count}")
            print(f"Agent execution time: {elapsed_time:.2f} seconds")
            results = {
                "task_outputs": new_files,
                "tools_count": tool_usage_count,
                "thoughts_count": thoughts,
                "processing_time": elapsed_time,
            }


            # Append results to JSON file
            log_file = "agent_results.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as file:
                    try:
                        existing_results = json.load(file)
                    except json.JSONDecodeError:
                        existing_results = {}
            else:
                existing_results = {}

            task_name = f"task_{len(existing_results) + 1}"
            existing_results[task_name] = results

            with open(log_file, "w") as file:
                json.dump(existing_results, file, indent=4)

            print(f"Results logged to {log_file}")

@xai_component
class ConvertImageFormat(Component):
    """
    This component converts a batch of images from one format to another and saves them in a specified output directory.

    ##### inPorts:
    - input_directory: Path to the directory containing the source images.
    - output_directory: Path to the directory where converted images will be saved.
    - input_format: The format of the source images (e.g., "png").
    - output_format: The desired format of the converted images (e.g., "jpeg").

    ##### outPorts:
    - converted_files: A list of paths to the converted images.
    - result: A summary of the conversion process.
    """

    input_directory: InArg[str]
    output_directory: InArg[str]
    input_format: InArg[str]
    output_format: InArg[str]

    converted_files: OutArg[list]
    result: OutArg[str]

    def execute(self, ctx) -> None:
        import os
        from PIL import Image

        input_dir = self.input_directory.value
        output_dir = self.output_directory.value
        input_format = self.input_format.value.lower()
        output_format = self.output_format.value.lower()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        converted_paths = []
        total_files = 0
        successful_conversions = 0
        failed_conversions = 0

        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(f".{input_format}"):
                total_files += 1
                input_path = os.path.join(input_dir, file_name)

                try:
                    # Open and convert the image
                    img = Image.open(input_path)
                    output_file_name = file_name.rsplit(f".{input_format}", 1)[0] + f".{output_format}"
                    output_path = os.path.join(output_dir, output_file_name)

                    img.convert("RGB").save(output_path, output_format.upper())
                    converted_paths.append(output_path)
                    successful_conversions += 1
                except Exception as e:
                    print(f"Failed to convert {input_path}: {e}")
                    failed_conversions += 1

        self.converted_files.value = converted_paths
        self.result.value = (
            f"Conversion Summary: Successfully converted {successful_conversions} out of {total_files} images. "
            f"Failed conversions: {failed_conversions}."
        )

@xai_component()
class ExtractImageConversionDetails(Component):
    """
    Component to extract input_directory, output_directory, input_format, and output_format from a JSON string.
    """
    input_json: InArg[str]

    input_directory: OutArg[str]
    output_directory: OutArg[str]
    input_format: OutArg[str]
    output_format: OutArg[str]

    def execute(self, ctx) -> None:
        import json

        # Parse the input JSON string
        input_data = json.loads(self.input_json.value)

        # Extract values
        input_directory = input_data.get('input_directory', '')
        output_directory = input_data.get('output_directory', '')
        input_format = input_data.get('input_format', '')
        output_format = input_data.get('output_format', '')

        # Set outputs
        self.input_directory.value = str(input_directory)
        self.output_directory.value = str(output_directory)
        self.input_format.value = str(input_format)
        self.output_format.value = str(output_format)

        # Print for debugging
        print(self.input_directory.value, self.output_directory.value, self.input_format.value, self.output_format.value)

@xai_component
class WriteFinancialAnalysisReport(Component):
    """
    Generates a financial analysis report based on the data in the documents at the specified folder path.

    ##### inPorts:
    - folder_path: Path to the folder containing financial documents.

    ##### outPorts:
    - report: The generated financial analysis report in JSON format.
    """

    folder_path: InArg[str]
    report: OutArg[str]

    def read_financial_documents(self, folder_path):
        data_frames = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                print(f"Skipping unsupported file: {file_name}")
                continue
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def analyze_financial_data(self, data):
        analysis = {
            'total_revenue': data['Revenue'].sum(),
            'total_expenses': data['Expenses'].sum(),
            'net_profit': data['Revenue'].sum() - data['Expenses'].sum(),
            'average_revenue': data['Revenue'].mean(),
            'average_expenses': data['Expenses'].mean(),
            'average_customer_satisfaction': data['Customer Satisfaction (%)'].mean(),
        }

        department_analysis = {}
        for department in data['Department'].unique():
            dept_data = data[data['Department'] == department]
            department_analysis[department] = {
                'total_revenue': dept_data['Revenue'].sum(),
                'total_expenses': dept_data['Expenses'].sum(),
                'net_profit': dept_data['Revenue'].sum() - dept_data['Expenses'].sum(),
                'average_revenue': dept_data['Revenue'].mean(),
                'average_expenses': dept_data['Expenses'].mean(),
            }
        analysis['department_analysis'] = department_analysis

        region_analysis = {}
        for region in data['Region'].unique():
            region_data = data[data['Region'] == region]
            region_analysis[region] = {
                'total_revenue': region_data['Revenue'].sum(),
                'total_expenses': region_data['Expenses'].sum(),
                'net_profit': region_data['Revenue'].sum() - region_data['Expenses'].sum(),
                'average_revenue': region_data['Revenue'].mean(),
                'average_expenses': region_data['Expenses'].mean(),
            }
        analysis['region_analysis'] = region_analysis

        analysis['total_transactions'] = data['Number of Transactions'].sum()
        analysis['average_transactions'] = data['Number of Transactions'].mean()

        return analysis

    def default_converter(self, o):
        if isinstance(o, (pd.Timestamp, pd.Timedelta)):
            return str(o)
        elif isinstance(o, (int, float)):  # Handle numeric types directly
            return o
        elif pd.isna(o):  # Handle NaN values
            return None
        else:
            return str(o)  # Convert any unsupported type to string

    def execute(self, ctx) -> None:
        folder_path = self.folder_path.value

        # Read documents
        data = self.read_financial_documents(folder_path)

        # Analyze data
        analysis = self.analyze_financial_data(data)

        # Convert analysis to JSON format
        report_content = json.dumps(analysis, indent=4, default=self.default_converter)

        # Set output
        self.report.value = report_content



@xai_component()
class ExtractDetails(Component):
    """
    Component to extract folder_path and outpuy_folder_name from a JSON string.
    """
    input_json: InArg[str]

    folder_path: OutArg[str]



    def execute(self, ctx) -> None:
        import json

        # Parse the input JSON string
        input_data = json.loads(self.input_json.value)

        # Extract values
        folder_path = input_data.get('folder_path', '')
        input_format = input_data.get('input_format', '')
        output_format = input_data.get('output_format', '')

        # Set outputs
        self.folder_path.value = str(folder_path)


        # Print for debugging
        print(self.folder_path.value)

@xai_component
class TranslateFile(Component):
    """
    Component to translate the content of a text file to a specified target language.

    ##### inPorts:
    - input_file: Path to the input text file to be translated.
    - target_language: The language to translate the text into (e.g., "ar" for Arabic, "fr" for French).

    ##### outPorts:
    - translated_text: The translated text content.
    """

    input_file: InArg[str]
    target_language: InArg[str]

    translated_text: OutArg[str]

    def execute(self, ctx) -> None:
        from translate import Translator

        translator = Translator(to_lang=self.target_language.value)
        input_file_path = self.input_file.value

        try:
            # Read the content of the input file
            with open(input_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Translate the content
            translated_content = translator.translate(content)

            # Set the translated text as output
            self.translated_text.value = translated_content
            print("Translation successful.")
        except Exception as e:
            print(f"Translation failed: {e}")
            self.translated_text.value = f"Error: {e}"

@xai_component
class SaveTextFile(Component):
    """
    Saves a given text to a file in the specified format and directory, then returns the file path.

    ##### inPorts:
    - text: The text to save.
    - file_name: The desired name of the file (without extension).
    - file_format: The format in which the file should be saved (e.g., 'txt', 'json', 'html').
    - save_directory: The directory where the file should be saved.

    ##### outPorts:
    - saved_file_path: The path to the saved file.
    """
    text: InArg[str]
    file_name: InArg[str]
    file_format: InArg[str]
    save_directory: InArg[str]

    saved_file_path: OutArg[str]

    def execute(self, ctx) -> None:
        import os
        import json

        # Input values
        text = self.text.value
        file_name = self.file_name.value
        file_format = self.file_format.value.lower()
        save_directory = self.save_directory.value

        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Generate full file path
        file_path = os.path.join(save_directory, f"{file_name}.{file_format}")

        try:
            if file_format == "txt":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            elif file_format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({"content": text}, f, ensure_ascii=False, indent=4)
            elif file_format == "html":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"<html><body><pre>{text}</pre></body></html>")
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            self.saved_file_path.value = file_path
            print(f"File saved successfully at: {file_path}")

        except Exception as e:
            print(f"Error saving file: {e}")
            self.saved_file_path.value = None

@xai_component()
class ExtractTranslateDetails(Component):
    """
    Component to extract folder_path and outpuy_folder_name from a JSON string.
    """
    input_json: InArg[str]

    input_file: OutArg[str]
    target_language: OutArg[str]


    def execute(self, ctx) -> None:
        import json

        # Parse the input JSON string
        input_data = json.loads(self.input_json.value)

        # Extract values
        input_file = input_data.get('input_file', '')
        target_language = input_data.get('target_language', '')

        # Set outputs
        self.input_file.value = str(input_file)
        self.target_language.value = str(target_language)


        # Print for debugging
        print(self.input_file.value, self.target_language.value)


@xai_component()
class ExtractSaveDetails(Component):
    """
    Component to extract text, file_name, file_format, and save_directory from a JSON string.
    """
    input_json: InArg[str]

    text: OutArg[str]
    file_name: OutArg[str]
    file_format: OutArg[str]
    save_directory: OutArg[str]

    def execute(self, ctx) -> None:
        import json

        # Parse the input JSON string
        input_data = json.loads(self.input_json.value)

        # Extract values
        text = input_data.get('text', '')
        file_name = input_data.get('file_name', '')
        file_format = input_data.get('file_format', '')
        save_directory = input_data.get('save_directory', '')

        # Set outputs
        self.text.value = str(text)
        self.file_name.value = str(file_name)
        self.file_format.value = str(file_format)
        self.save_directory.value = str(save_directory)

        # Print for debugging
        print(self.text.value, self.file_name.value, self.file_format.value, self.save_directory.value)


@xai_component
class GenerateFinancialCharts(Component):
    """
    Component to generate financial charts from the final financial report in JSON format.

    ##### inPorts:
    - report_data: The financial data as a JSON string.
    - output_folder: The folder where the charts will be saved.

    ##### outPorts:
    - charts_paths: A JSON string containing file paths for the generated charts.
    """

    report_data: InArg[str]
    output_folder: InArg[str]
    charts_paths: OutArg[str]  # Changed from list to str for JSON output

    def execute(self, ctx) -> None:

        # Read inputs
        report_data = self.report_data.value
        output_folder = self.output_folder.value

        try:
            # Parse the JSON string
            data = json.loads(report_data)

            # Convert top-level numeric fields
            for key in ['total_revenue', 'total_expenses', 'net_profit']:
                if isinstance(data[key], str):
                    data[key] = float(data[key])

            # Convert department and region analyses to DataFrames
            department_df = pd.DataFrame.from_dict(data['department_analysis'], orient='index')
            region_df = pd.DataFrame.from_dict(data['region_analysis'], orient='index')

            # Convert columns to numeric where applicable
            department_df = department_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
            region_df = region_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)

            charts = []

            # Total Revenue vs Expenses
            plt.figure()
            pd.DataFrame({
                'Total Revenue': [data['total_revenue']],
                'Total Expenses': [data['total_expenses']]
            }).plot(kind='bar', rot=0)
            plt.title('Total Revenue vs Expenses')
            plt.ylabel('Amount')
            chart_path = os.path.join(output_folder, 'total_revenue_vs_expenses.png')
            plt.savefig(chart_path)
            plt.close()
            charts.append(chart_path)

            # Net Profit by Department
            if 'net_profit' in department_df.columns:
                plt.figure()
                department_df['net_profit'].plot(kind='bar')
                plt.title('Net Profit by Department')
                plt.ylabel('Net Profit')
                chart_path = os.path.join(output_folder, 'net_profit_by_department.png')
                plt.savefig(chart_path)
                plt.close()
                charts.append(chart_path)
            else:
                print("Column 'net_profit' is missing in department data.")

            # Regional Performance (Net Profit)
            if 'net_profit' in region_df.columns:
                plt.figure()
                region_df['net_profit'].plot(kind='pie', autopct='%1.1f%%')
                plt.title('Net Profit by Region')
                chart_path = os.path.join(output_folder, 'regional_performance.png')
                plt.savefig(chart_path)
                plt.close()
                charts.append(chart_path)
            else:
                print("Column 'net_profit' is missing in region data.")

            # Convert the charts list to JSON
            charts_json = json.dumps({"charts_paths": charts}, indent=4)

            # Set JSON output
            self.charts_paths.value = charts_json
            print("Charts generated successfully.")
        except KeyError as e:
            print(f"Missing key in JSON data: {e}")
            self.charts_paths.value = json.dumps({"error": f"Missing key: {e}"})
        except ValueError as e:
            print(f"Value error in data: {e}")
            self.charts_paths.value = json.dumps({"error": f"Value error: {e}"})
        except Exception as e:
            print(f"Failed to generate charts: {e}")
            self.charts_paths.value = json.dumps({"error": f"Exception: {e}"})

@xai_component()
class ExtractChartDetails(Component):
    """
    Component to extract report data and output folder path from a JSON string.
    """
    input_json: InArg[str]

    report_data: OutArg[str]
    output_folder: OutArg[str]

    def execute(self, ctx) -> None:
        import json

        # Parse the input JSON string
        input_data = json.loads(self.input_json.value)

        # Extract values
        report_data = input_data.get('report_data', '')
        output_folder = input_data.get('output_folder', '')

        # Set outputs
        self.report_data.value = str(report_data)
        self.output_folder.value = str(output_folder)

        # Print for debugging
        print(f"Report Data: {self.report_data.value}")
        print(f"Output Folder: {self.output_folder.value}")

@xai_component
class OpenAIAuthorize(Component):
    """Sets the organization and API key for the OpenAI client and creates an OpenAI client.

    This component checks if the API key should be fetched from the environment variables or from the provided input.
    It then creates an OpenAI client using the API key and stores the client in the context (`ctx`) for use by other components.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/authentication)

    ##### inPorts:
    - organization: Organization name id for OpenAI API.
    - api_key: API key for the OpenAI API.
    - from_env: Boolean value indicating whether the API key is to be fetched from environment variables.

    """
    organization: InArg[secret]
    base_url: InArg[str]
    api_key: InArg[secret]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        openai.organization = self.organization.value
        openai.base_url= self.base_url.value
        if self.from_env.value:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = self.api_key.value

        client = OpenAI(api_key=self.api_key.value)
        ctx['client'] = client
        ctx['openai_api_key'] = openai.api_key

@xai_component
class SlackClient(Component):
    """
    A component that initializes a Slack WebClient with the provided `slack_bot_token`. The created client is then added to the context for further use by other components.

    ## Inputs
    - `slack_bot_token` (optional): The Slack bot token used to authenticate the WebClient. If not provided, it will try to read the token from the environment variable `SLACK_BOT_TOKEN`.

    ## Outputs
    - Adds the `slack_client` object to the context for other components to use.
    """
    slack_bot_token: InArg[secret]

    def execute(self, ctx) -> None:
        slack_bot_token = os.getenv("SLACK_BOT_TOKEN") if self.slack_bot_token.value is None else self.slack_bot_token.value
        slack_client = WebClient(slack_bot_token)
        ctx.update({'slack_client':slack_client})

@xai_component
class SlackSendMessageToServerAndChannel(Component):
    """
    A component that sends a message to a specific server and channel in Slack.

    ## Inputs
    - `server_url`: The URL of the server to send the message to.
    - `channel_id`: The ID of the channel in the server to send the message to.
    - `message`: The message content to be sent.

    ## Requirements
    - `slack_client` instance in the context (created by `SlackClient` component).
    """
    server_url: InArg[str]
    channel_id: InArg[str]
    message: InArg[str]

    def execute(self, ctx) -> None:
        slack_client = ctx['slack_client']
        server_url = self.server_url.value
        channel_id = self.channel_id.value
        message = self.message.value

        response = slack_client.chat_postMessage(channel=channel_id, text=message)
        print(f"Message sent to server {server_url} and channel {channel_id}: {message}")
