from crewai import Agent, LLM
from tools import TranslationTools,SaveTextFileTools, ConversionTools, FinancialAnalysisTools, VisualizationTools, SlackTools
from langchain_openai import ChatOpenAI

class TranslatorAgents:
    def translator_agent(self):
        chat_llm = ChatOpenAI(
            model_name="gpt-4o",  
            temperature=0.7
        )
        
        return Agent(
            role='Translator, Image Converter, Financial Analyst, and Chart Generator',
            goal='Translate texts, convert images, analyze financial data, and generate financial charts.',
            backstory='A multi-functional agent capable of handling various tasks.',
            tools=[
                TranslationTools.translate_text,
                SaveTextFileTools.save_text_file,
                ConversionTools.convert_images,
                FinancialAnalysisTools.analyze_financial_data,
                VisualizationTools.generate_charts,
                SlackTools.send_summary_to_slack
            ],
            llm=chat_llm,
        )
