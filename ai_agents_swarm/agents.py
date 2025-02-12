# agents.py
import os
import json
import pandas as pd
from PIL import Image
import deepl
from slack_sdk import WebClient
from swarm import Agent

def translate_text(input_file: str, target_language: str) -> str:
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        auth_key = os.getenv("DEEPL_API_KEY")
        if not auth_key:
            return "Error: DEEPL_API_KEY not found."
        
        translator = deepl.Translator(auth_key)
        result = translator.translate_text(text, target_lang=target_language.upper())
        
        return result.text
    
    except Exception as e:
        return f"Translation error: {str(e)}"

    
def write_file_content(file_path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File successfully saved: {file_path}"
    except Exception as e:
        return f"Error saving file {file_path}: {str(e)}"    
  
def convert_images(input_folder: str, output_folder: str, target_format: str) -> str:
    try:
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.{target_format}")
                with Image.open(input_path) as img:
                    img.convert("RGB").save(output_path, target_format.upper())
        return f"Images converted to {target_format.upper()} and saved in {output_folder}"
    except Exception as e:
        return f"Image conversion error: {str(e)}"

def analyze_financial_data(folder_path: str) -> dict:
    """
    Reads financial data from a specified folder, performs analysis, 
    and returns a structured report in JSON format.

    Arguments:
    - folder_path: Path to the folder containing financial CSV/XLSX files.

    Returns:
    - A dictionary containing a text summary and a JSON object with the detailed analysis.
    """

    def read_financial_documents(folder_path):
        """Reads all CSV/XLSX files in a given folder and combines them into a single DataFrame."""
        data_frames = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                continue
            data_frames.append(df)

        if not data_frames:
            raise ValueError("No valid financial data files found in the specified folder.")

        return pd.concat(data_frames, ignore_index=True)

    def analyze_data(data):
        """Performs financial analysis on the given DataFrame."""
        import numpy as np

        analysis = {
            'total_revenue': int(data['Revenue'].sum()),
            'total_expenses': int(data['Expenses'].sum()),
            'net_profit': int(data['Revenue'].sum() - data['Expenses'].sum()),
            'average_revenue': float(data['Revenue'].mean()),
            'average_expenses': float(data['Expenses'].mean()),
            'average_customer_satisfaction': float(data['Customer Satisfaction (%)'].mean()),
        }

        department_analysis = {}
        for department in data['Department'].unique():
            dept_data = data[data['Department'] == department]
            department_analysis[department] = {
                'total_revenue': int(dept_data['Revenue'].sum()),
                'total_expenses': int(dept_data['Expenses'].sum()),
                'net_profit': int(dept_data['Revenue'].sum() - dept_data['Expenses'].sum()),
            }
        analysis['department_analysis'] = department_analysis

        region_analysis = {}
        for region in data['Region'].unique():
            region_data = data[data['Region'] == region]
            region_analysis[region] = {
                'total_revenue': int(region_data['Revenue'].sum()),
                'total_expenses': int(region_data['Expenses'].sum()),
                'net_profit': int(region_data['Revenue'].sum() - region_data['Expenses'].sum()),
            }
        analysis['region_analysis'] = region_analysis

        analysis['total_transactions'] = int(data['Number of Transactions'].sum())
        analysis['average_transactions'] = float(data['Number of Transactions'].mean())

        return analysis

    try:
        data = read_financial_documents(folder_path)
        analysis = analyze_data(data)

        report_text = f"""
         **Financial Analysis Report** 

        **Total Revenue:** {analysis['total_revenue']}
        **Total Expenses:** {analysis['total_expenses']}
        **Net Profit:** {analysis['net_profit']}
        **Average Revenue:** {analysis['average_revenue']:.2f}
        **Average Expenses:** {analysis['average_expenses']:.2f}
        **Average Customer Satisfaction:** {analysis['average_customer_satisfaction']:.2f}%

         **Department Analysis:**
        """ + "\n".join(
            f"- {dept}: Revenue: {stats['total_revenue']}, Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}"
            for dept, stats in analysis['department_analysis'].items()
        ) + f"""

         **Region Analysis:**
        """ + "\n".join(
            f"- {region}: Revenue: {stats['total_revenue']}, Expenses: {stats['total_expenses']}, Net Profit: {stats['net_profit']}"
            for region, stats in analysis['region_analysis'].items()
        ) + f"""

         **Total Transactions:** {analysis['total_transactions']}
         **Average Transactions:** {analysis['average_transactions']:.1f}
        """

        return {"text_report": report_text, "json_report": analysis}

    except Exception as e:
        return {"error": f"An error occurred during analysis: {str(e)}"}

import os
import json
import matplotlib.pyplot as plt

def generate_charts(input_json: str, output_folder: str) -> str:
    """
    Generates financial charts based on JSON financial data.

    Arguments:
    - input_json: Path to the JSON file containing financial analysis data.
    - output_folder: Path to the folder where the charts will be saved.

    Returns:
    - A confirmation message indicating success or failure.
    """

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
            return "Error: `department_analysis` or `region_analysis` is missing or empty."

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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

        plt.figure(figsize=(10, 6))
        regions = list(region_analysis.keys())
        revenues = [int(region_analysis[region]['total_revenue']) for region in regions]
        expenses = [int(region_analysis[region]['total_expenses']) for region in regions]
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

        plt.figure(figsize=(10, 6))
        departments = list(department_analysis.keys())
        net_profits = [int(department_analysis[dept]['net_profit']) for dept in departments]
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


def send_summary_to_slack(message: str) -> str:
    try:
        slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        slack_channel = os.getenv("SLACK_CHANNEL_ID")
        if not slack_bot_token or not slack_channel:
            return "Slack configuration missing."
        client = WebClient(token=slack_bot_token)
        response = client.chat_postMessage(channel=slack_channel, text=message)
        if response.get("ok"):
            return f"Summary sent to Slack channel {slack_channel}"
        else:
            return f"Slack error: {response.get('error')}"
    except Exception as e:
        return f"Slack error: {str(e)}"

multi_agent = Agent(
    name="MultiToolAgent",
    instructions="Agent that translates text, converts images, analyzes financial data, and sends summaries to Slack.",
    functions=[translate_text, convert_images, analyze_financial_data, send_summary_to_slack,write_file_content, generate_charts]
)
