"""
Configuration settings for the Academic Paper Generator
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Configuration (supports both OpenAI and Azure OpenAI)
    USE_AZURE_OPENAI = os.getenv('USE_AZURE_OPENAI', 'false').lower() == 'true'
    
    # Standard OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY', '')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')
    MODEL_DEPLOYMENT_NAME = os.getenv('MODEL_DEPLOYMENT_NAME', 'gpt-4-mini')
    
    # Paper generation settings
    MAX_PAPERS_TO_RETRIEVE = 20
    TARGET_PAPER_LENGTH = 2000  # words for 2 pages
    CITATION_SIMILARITY_THRESHOLD = 0.7
    
    # PDF processing settings
    MAX_PDF_SIZE_MB = 50
    PDF_DOWNLOAD_TIMEOUT = 30
    
    # Content generation settings (model name depends on provider)
    OPENAI_MODEL = MODEL_DEPLOYMENT_NAME if USE_AZURE_OPENAI else "gpt-4"
    MAX_TOKENS_PER_REQUEST = 4000
    TEMPERATURE = 0.3  # Lower for more factual content
    
    # Output settings
    OUTPUT_DIR = "generated_papers"
    INCLUDE_GRAPHS = True
    INCLUDE_DATA_ANALYSIS = True
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.USE_AZURE_OPENAI:
            if not cls.AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY is required when using Azure OpenAI")
            if not cls.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when using Azure OpenAI")
        else:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when using standard OpenAI")
        return True
    
    @classmethod
    def validate_configuration(cls):
        """Validate that required environment variables are set"""
        if cls.USE_AZURE_OPENAI:
            if not cls.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required when using Azure OpenAI")
            if not cls.AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is required when using Azure OpenAI")
        else:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is required when using standard OpenAI")

    @classmethod
    def get_openai_client(cls):
        """Get the appropriate OpenAI client based on configuration"""
        cls.validate_configuration()
        
        if cls.USE_AZURE_OPENAI:
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT,
                api_key=cls.AZURE_OPENAI_API_KEY,
                api_version=cls.AZURE_OPENAI_API_VERSION
            )
        else:
            import openai
            openai.api_key = cls.OPENAI_API_KEY
            return openai
