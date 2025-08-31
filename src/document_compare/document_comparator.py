import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.customexception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY
import pandas as pd

class Documentcomparator:
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.model_loader=ModelLoader()
            self.llm = self.model_loader.load_llm()

            self.prompt = PROMPT_REGISTRY['document_comparison']
            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("DocumentComparator initialized successfully")

        except Exception as e:
            self.log.error("Failed to initialize DocumentComparator", error=str(e))
            raise DocumentPortalException("Failed to initialize DocumentComparator", sys) from e


    def compare_documents(self, combined_documents:str) -> pd.DataFrame:

        try:
            

            inputs={"combined_docs": combined_documents,
            "format_instruction": self.parser.get_format_instructions()
            
            
            }  

            self.log.info("Starting document comparison", inputs=inputs)


            response = self.chain.invoke(inputs)

            self.log.info("Document comparison successful")

            return self.format_response(response)
        

        except Exception as e:
            self.log.error("Failed to compare documents", error=str(e))
            raise DocumentPortalException("Failed to compare documents", sys) from e
        

    def format_response(self, response: list[dict]) -> pd.DataFrame:

        try:
            

            df = pd.DataFrame.from_dict(response)
            

            

            return df

        except Exception as e:

            self.log.error("Error formatting response into DataFrame", error=str(e))
            DocumentPortalException("Error formatting response", sys)
          
        
