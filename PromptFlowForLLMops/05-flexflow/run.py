import os
from promptflow.tracing import start_trace, trace
from dotenv import load_dotenv
from promptflow.core import Prompty, AzureOpenAIModelConfiguration

#starting the trace on our python chat function
@trace
def start_chat_with_model(input: str) -> str:
    #loading the .env file
    load_dotenv()
    #creating the AzureOpenAIModelConfiguration object
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_KEY"),
        azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        api_version="2024-05-01-preview"
    )

    #creating the Prompty object
    prompty_file_path = r'chat.prompty'
    # prompty_file_path = r'PromptFlowForLLMops\05-flexflow\chat.prompty'
    prompty = Prompty.load(prompty_file_path, model={'configuration': model_config})

    #prompting the user with the chat history and chat input
    result=prompty(chat_input=input)
    
    #displaying the response
    return result

if __name__ == "__main__":
    start_trace()
    result = start_chat_with_model("Hello, how are you?")
    print(result)