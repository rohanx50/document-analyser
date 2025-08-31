import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
#from .config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from logger.custom_logger import CustomLogger
from exception.customexception import DocumentPortalException
log = CustomLogger().get_logger(__name__)


class ModelLoader:
    def __init__(self):
       load_dotenv()
       self._validate_environment()
       self.config = load_config()
       
       log.info("Configuration loaded successfully", config=self.config)

    def _validate_environment(self):
        required_env_vars = [  "OPENAI_API_KEY","GOOGLE_API_KEY"]
        self.api_keys = {var: os.getenv(var) for var in required_env_vars   }
        missing_vars = [var for var, value in self.api_keys.items() if not value]
        if missing_vars:
           log.error(f"Missing environment variables: {', '.join(missing_vars)}")
           raise DocumentPortalException(f"Missing environment variables: {', '.join(missing_vars)}",sys)   
        
        log.info("Environment variables validated successfully")

      

    def load_embeddings(self):

        try:
            log.info('loading embeddings')
            model_name=self.config['embedding_model']['model_name']

            return GoogleGenerativeAIEmbeddings(model=model_name,google_api_key=self.api_keys.get('GOOGLE_API_KEY'))
            
        
        
        except Exception as e:
            log.error("Failed to load embeddings", error=str(e))
            raise DocumentPortalException("Failed to load embeddings", sys) 
        

    def load_llm(self):
        provider=os.getenv("LLM_PROVIDER", "openai")

        llm_block = self.config['llm']
        if provider not in llm_block:
            log.error(f"LLM provider '{provider}' not found in configuration")
            raise DocumentPortalException(f"LLM provider '{provider}' not found in configuration", sys)
        

        if provider == "google":
            model_name = llm_block[provider].get('model_name', 'gemini-2.0-flash')
            temperature = llm_block[provider].get('temperature', 0)
            max_output_tokens = llm_block[provider].get('max_output_tokens', 2048)

            log.info(f"Loading Google LLM: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_output_tokens)   
        

        elif provider == "groq":
            model_name = llm_block[provider].get('model_name', 'groq-llama-3.1')
            temperature = llm_block[provider].get('temperature', 0)
            max_output_tokens = llm_block[provider].get('max_output_tokens', 2048)

            log.info(f"Loading Groq LLM: {model_name}")
            return ChatGroq(model=model_name, temperature=temperature, max_output_tokens=max_output_tokens)
        

        elif provider == "openai":
            model_name = llm_block[provider].get('model_name', 'gpt-4o')
            temperature = llm_block[provider].get('temperature', 0)
            max_output_tokens = llm_block[provider].get('max_output_tokens', 2048)
            api_key = self.api_keys.get('OPENAI_API_KEY')

            log.info(f"Loading OpenAI LLM: {model_name}")
            return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_output_tokens,api_key=api_key)
        
        else:
            log.error(f"Unsupported LLM provider: {provider}")
            raise DocumentPortalException(f"Unsupported LLM provider: {provider}", sys)
        



if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test embedding model loading
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    
    # Test the ModelLoader
    result=embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")



               

            


    
    
