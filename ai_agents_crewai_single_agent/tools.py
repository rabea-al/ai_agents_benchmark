from PIL import Image
import os
from langchain_community.tools import tool
import deepl
import pandas as pd
import json
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import os
import deepl
from langchain_community.tools import tool

class TranslationTools:
    """
    A toolset for translating text using DeepL API.
    
    Features:
    - Reads text from a file automatically if a file path is provided.
    - Supports direct text input for translation.
    - Handles file reading errors and missing API keys gracefully.
    """

    name = "TranslationTools"
    description = "Provides translation capabilities using DeepL API."

    @tool("Translate text")
    def translate_text(input_source: str, target_language: str):
        """
        Translates text from a file or a direct string into the specified language using DeepL API.

        Parameters:
        - input_source (str): Either a file path containing text or a direct text string.
        - target_language (str): The target language code (e.g., "en" for English, "ja" for Japanese).

        Returns:
        - Translated text or an error message if an issue occurs.
        """

        def read_text(file_path: str) -> str:
            """Reads the content of a file if it exists, otherwise returns an error message."""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return f"Error: The file '{file_path}' was not found."
            except Exception as e:
                return f"Error reading file '{file_path}': {str(e)}"

        # Determine if input_source is a file path or direct text
        text = read_text(input_source) if os.path.isfile(input_source) else input_source

        # Ensure API key is available
        auth_key = os.getenv("DEEPL_API_KEY")
        if not auth_key:
            return "Error: DeepL API key is missing."

        # Perform translation
        try:
            translator = deepl.Translator(auth_key)
            translated_text = translator.translate_text(text, target_lang=target_language.upper())
            return translated_text.text
        except Exception as e:
            return f"Error during translation: {str(e)}"



    
class SaveTextFileTools:
    name = "SaveTextFileTools"
    description = "Tool for saving text content to files in various formats (txt, json, html)."

    @tool("Save text file")
    def save_text_file(text: str, file_name: str, file_format: str, save_directory: str):
        """
        Saves the provided text into a file with the specified name, format, and directory.
        Supported file formats: txt, json, html.
        """
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)
        file_format = file_format.lower()
        
        if file_name.lower().endswith(f".{file_format}"):
            file_path = os.path.join(save_directory, file_name)
        else:
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
                return f"Error: Unsupported file format '{file_format}'. Supported formats: txt, json, html."

            return f"File saved successfully at: {file_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"



class ConversionTools:
    name = "ConversionTools"
    description = "Tool for converting image formats."

    @tool("Convert images")
    def convert_images(description: str):
        """
        Extracts conversion parameters from the JSON-formatted description,
        then converts images in the specified folder to the target format.

        Expected JSON format:
        {
            "input_folder": "source_directory",
            "output_folder": "destination_directory",
            "target_format": "png"  // or "jpg", "jpeg", "bmp", "gif"
        }

        Returns:
        - A confirmation message with the output folder.
        """
        try:
            params = json.loads(description)
        except json.JSONDecodeError:
            return "Error: Description is not a valid JSON string with conversion parameters."

        input_folder = params.get("input_folder")
        output_folder = params.get("output_folder")
        target_format = params.get("target_format")

        if not input_folder or not output_folder or not target_format:
            return "Error: Missing one or more required parameters: input_folder, output_folder, target_format."

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                input_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.{target_format}")
                try:
                    with Image.open(input_path) as img:
                        img.convert("RGB").save(output_path, target_format.upper())
                except Exception as e:
                    print(f"Error converting {filename}: {str(e)}")
        return f"All images converted to {target_format.upper()} and saved in {output_folder}"

class FinancialAnalysisTools:
    name = "FinancialAnalysisTools"
    description = "Reads financial data from documents and returns an analysis report."

    @tool("Analyze financial data")
    def analyze_financial_data(description: str):
        """
        Extracts financial analysis parameters from the JSON-formatted description,
        then reads financial data from documents and returns a detailed analysis report.

        Expected JSON format:
        {
            "folder_path": "path_to_financial_data"
        }

        Returns:
        - A text summary report.
        - A JSON object containing detailed financial analysis.
        """
        try:
            params = json.loads(description)
        except json.JSONDecodeError:
            return "Error: Description is not a valid JSON string with financial analysis parameters."

        folder_path = params.get("folder_path")
        if not folder_path:
            return "Error: Missing required parameter: folder_path."

        def read_financial_documents(folder_path):
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

        def analyze_data(data):
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
                }
            analysis['department_analysis'] = department_analysis

            region_analysis = {}
            for region in data['Region'].unique():
                region_data = data[data['Region'] == region]
                region_analysis[region] = {
                    'total_revenue': region_data['Revenue'].sum(),
                    'total_expenses': region_data['Expenses'].sum(),
                    'net_profit': region_data['Revenue'].sum() - region_data['Expenses'].sum(),
                }
            analysis['region_analysis'] = region_analysis

            analysis['total_transactions'] = data['Number of Transactions'].sum()
            analysis['average_transactions'] = data['Number of Transactions'].mean()
            return analysis

        try:
            data = read_financial_documents(folder_path)
            analysis = analyze_data(data)
            report_text = f"""
            Total Revenue: {analysis['total_revenue']}
            Total Expenses: {analysis['total_expenses']}
            Net Profit: {analysis['net_profit']}
            Average Revenue: {analysis['average_revenue']:.2f}
            Average Expenses: {analysis['average_expenses']:.2f}
            Average Customer Satisfaction: {analysis['average_customer_satisfaction']:.4f}

            Department Analysis:
            """ + "\n".join(
                f"- {dept}: Total Revenue: {stats['total_revenue']}, Total Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}"
                for dept, stats in analysis['department_analysis'].items()
            ) + "\n\nRegion Analysis:\n" + "\n".join(
                f"- {region}: Total Revenue: {stats['total_revenue']}, Total Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}"
                for region, stats in analysis['region_analysis'].items()
            ) + f"""

            Total Transactions: {analysis['total_transactions']}
            Average Transactions: {analysis['average_transactions']:.1f}
            """
            return {"text_report": report_text, "json_report": analysis}
        except Exception as e:
            return {"error": f"An error occurred during analysis: {str(e)}"}


class VisualizationTools:
    name = "VisualizationTools"
    description = "Generate charts from JSON financial data."

    @tool("Generate financial charts")
    def generate_charts(description: str):
        """
        Extracts chart generation parameters from the JSON-formatted description,
        then generates financial charts based on JSON financial data.

        Expected JSON format:
        {
            "input_json": "path_to_json_financial_data",
            "output_folder": "path_to_output_folder"
        }

        Returns:
        - A confirmation message with the output folder.
        """
        try:
            params = json.loads(description)
        except json.JSONDecodeError:
            return "Error: Description is not a valid JSON string with chart generation parameters."

        input_json = params.get("input_json")
        output_folder = params.get("output_folder")

        if not input_json or not output_folder:
            return "Error: Missing one or more required parameters: input_json, output_folder."

        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if "content" in raw_data:
                data = json.loads(raw_data["content"])
            else:
                data = raw_data

            department_analysis = data.get("department_analysis", {})
            region_analysis = data.get("region_analysis", {})

            if not department_analysis or not region_analysis:
                return "Error: 'department_analysis' or 'region_analysis' is missing or empty."

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Chart for Total Revenue vs Total Expenses
            plt.figure(figsize=(8, 6))
            revenue = int(data["total_revenue"])
            expenses = int(data["total_expenses"])
            labels = ['Total Revenue', 'Total Expenses']
            values = [revenue, expenses]
            plt.bar(labels, values, color=['green', 'red'])
            plt.title('Total Revenue vs Total Expenses')
            plt.ylabel('Amount')
            plt.savefig(f"{output_folder}/total_revenue_vs_expenses.png")
            plt.close()

            # Chart for Regional Performance
            plt.figure(figsize=(10, 6))
            regions = list(region_analysis.keys())
            revenues = [region_analysis[region]['total_revenue'] for region in regions]
            expenses = [region_analysis[region]['total_expenses'] for region in regions]
            x = range(len(regions))
            plt.bar(x, revenues, width=0.4, label='Revenue', align='center', color='blue')
            plt.bar(x, expenses, width=0.4, label='Expenses', align='edge', color='orange')
            plt.xticks(x, regions)
            plt.title('Regional Performance')
            plt.xlabel('Regions')
            plt.ylabel('Amount')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_folder}/regional_performance.png")
            plt.close()

            # Chart for Net Profit by Department
            plt.figure(figsize=(10, 6))
            departments = list(department_analysis.keys())
            net_profits = [department_analysis[dept]['net_profit'] for dept in departments]
            plt.bar(departments, net_profits, color='purple')
            plt.title('Net Profit by Department')
            plt.xlabel('Departments')
            plt.ylabel('Net Profit')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/net_profit_by_department.png")
            plt.close()

            return f"Charts generated and saved to {output_folder}"
        except Exception as e:
            return f"An error occurred during chart generation: {str(e)}"


class SlackTools:
    name = "SlackTools"
    description = "Tools to send task summaries to Slack channels."

    @tool("Send task summary to Slack")
    def send_summary_to_slack(result_summary: str):
        """
        Sends a summary message to a specific Slack channel.

        Arguments:
        - result_summary: The summary of the task results.

        Returns:
        - A success message or an error if the operation fails.
        """
        server_url = os.getenv("SLACK_SERVER_URL")
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        slack_bot_token = os.getenv("SLACK_BOT_TOKEN")

        if not all([server_url, channel_id, slack_bot_token]):
            return "Missing Slack configuration in .env. Please set SLACK_SERVER_URL, SLACK_CHANNEL_ID, and SLACK_BOT_TOKEN."

        try:
            client = WebClient(token=slack_bot_token)
            message = f"*Task Summary:* {result_summary}"

            response = client.chat_postMessage(channel=channel_id, text=message)

            if response.get("ok"):
                return f"Summary successfully sent to Slack channel {channel_id}."
            else:
                return f"Failed to send message. Error: {response.get('error')}"
        except SlackApiError as e:
            return f"Slack API Error: {e.response['error']}"
        except Exception as e:
            return f"An error occurred while sending the message: {str(e)}"
