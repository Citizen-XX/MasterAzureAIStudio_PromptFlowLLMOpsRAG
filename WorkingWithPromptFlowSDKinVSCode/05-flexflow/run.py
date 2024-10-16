import os
from promptflow.tracing import start_trace, trace
import dotenv
from dotenv import load_dotenv
from dotenv import load_dotenv
from promptflow.core import Prompty, AzureOpenAIModelConfiguration

#starting the trace on our python chat function
@trace
def start_chat_with_model(input: str) -> str:
    #loading the .env file
    load_dotenv()
    #creating the AzureOpenAIModelConfiguration object
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_KEY"),
        azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        api_version="2024-05-01-preview"
    )

    #creating the Prompty object
    prompty = Prompty.load("chat.prompty", model={'configuration': model_config})

    #prompting the user with the chat history and chat input
    result=prompty(chat_input=input)
    
    #displaying the response
    return result
