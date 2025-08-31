import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.customexception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY


log = CustomLogger().get_logger(__name__)

class DataAnalysis:

    def __init__(self):
        self.log= CustomLogger().get_logger(__name__)
        try:
            self.model_loader = ModelLoader()
            
            self.llm = self.model_loader.load_llm()
            self.log.info("Model Loader initialized successfully")

            self.parser=JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = PROMPT_REGISTRY['document_analysis']

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error("Failed to initialize DocumentAnalyzer", error=str(e))
            raise DocumentPortalException("Failed to initialize DocumentAnalyzer", sys) from e
        

    def analyze_document(self, document_text: str) -> Metadata:

        """
        Analyze a document's text and extract structured metadata & summary.
        """

        try:
            chain= self.prompt |  self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })


            self.log.info("Metadata extraction successful", keys=list(response.keys()))

            return response


        except Exception as e:
            self.log.error("Failed to analyze document", error=str(e))
            raise DocumentPortalException("Failed to analyze document", sys) from e
        











        



