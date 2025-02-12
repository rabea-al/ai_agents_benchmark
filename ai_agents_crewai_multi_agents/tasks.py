import json
from crewai import Task
from langchain_openai import ChatOpenAI
from textwrap import dedent

from crewai import Task
from langchain_openai import ChatOpenAI
from tools import TranslationTools, ConversionTools, FinancialAnalysisTools,VisualizationTools

class TranslatorTasks:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7
        )

    def translation_task(self, agent, description):
        """
        """
        params = TranslationTools._parse_parameters_with_gpt(description, self.llm)
        if not params:
            raise ValueError("Failed to extract translation parameters.")
        
        return Task(
            description=description,
            agent=agent,
            expected_output=f"Translated content saved in {params['output_file']}"
        )

class ConversionTasks:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7
        )

    def convert_images_task(self, agent, description):
        """
        """
        params = ConversionTools._parse_parameters_with_gpt(description, self.llm)
        if not params:
            raise ValueError("Failed to extract conversion parameters.")
        
        return Task(
            description=description,
            agent=agent,
            expected_output=f"All images converted to {params['target_format'].upper()} and saved in {params['output_folder']}"
        )
class FinancialAnalysisTasks:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7
        )

    def analyze_financial_data_task(self, agent, description):
        """
       
        """
        params = FinancialAnalysisTools._parse_parameters_with_gpt(description, self.llm)
        if not params:
            raise ValueError("Failed to extract financial analysis parameters.")
        
        return Task(
            description=description,
            agent=agent,
            expected_output="Financial analysis completed and returned as text and JSON."
        )

class VisualizationTasks:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7
        )

    def generate_charts_task(self, agent, description):
        """
  
        """
        params = VisualizationTools._parse_parameters_with_gpt(description, self.llm)
        if not params:
            raise ValueError("Failed to extract chart generation parameters.")
        
        return Task(
            description=description,
            agent=agent,
            expected_output=f"Charts generated and saved in {params['output_folder']}"
        )

from crewai import Task
from langchain_openai import ChatOpenAI
from tools import SlackTools

class SlackTasks:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7
        )

    def send_summary_task(self, agent, description, result_summary):
        """

        """
        params = SlackTools._extract_slack_request(description, self.llm)
        if not params["send_to_slack"]:
            return None 

        return Task(
            description="Send task summary to Slack.",
            agent=agent,
            expected_output="Summary successfully sent to Slack."
        )
