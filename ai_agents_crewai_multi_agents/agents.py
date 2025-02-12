# agents.py
from crewai import Agent
from tools import (
    TranslationTools,
    SaveTextFileTools,
    ConversionTools,
    FinancialAnalysisTools,
    VisualizationTools,
    SlackTools,
)
from langchain_openai import ChatOpenAI

class TranslatorAgent:
    def get_agent(self):
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

        return Agent(
            role='Translator',
            goal=(
                "Translate the text and ensure that the translated questions are also answered. "
                "After translating, verify the output file to check if answers exist. "
                "If answers are missing, generate them using your reasoning abilities before finalizing the task."
            ),
            backstory=(
                "An AI agent specialized in translation tasks. "
                "It ensures all required steps are completed, including translating the content and answering any questions. "
                "If the output contains only translated questions, the agent will generate answers before saving."
            ),
            tools=[
                TranslationTools.translate_text,  
                SaveTextFileTools.save_text_file 
            ],
            llm=chat_llm, 
        )


class ConversionAgent:
    def get_agent(self):
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        return Agent(
            role='Image Converter',
            goal='Convert images from one format to another.',
            backstory='An agent specialized in image conversion tasks.',
            tools=[
                ConversionTools.convert_images
            ],
            llm=chat_llm,
        )

class FinancialAgent:
    def get_agent(self):
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        return Agent(
            role='Financial Analyst',
            goal='Analyze financial data and generate a report.',
            backstory='An agent specialized in financial analysis tasks.',
            tools=[
                FinancialAnalysisTools.analyze_financial_data,
                SaveTextFileTools.save_text_file
            ],
            llm=chat_llm,
        )

class VisualizationAgent:
    def get_agent(self):
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        return Agent(
            role='Chart Generator',
            goal='Generate charts from financial data.',
            backstory='An agent specialized in generating visualizations.',
            tools=[
                VisualizationTools.generate_charts
            ],
            llm=chat_llm,
        )

class SlackAgent:
    def get_agent(self):
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        return Agent(
            role='Slack Notifier',
            goal='Send task summaries to Slack.',
            backstory='An agent specialized in sending notifications via Slack.',
            tools=[
                SlackTools.send_summary_to_slack
            ],
            llm=chat_llm,
        )
