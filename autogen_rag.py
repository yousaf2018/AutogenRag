import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

llm_config = {"config_list": autogen.config_list_from_json("OAI_CONFIG_LIST")}

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant",
    llm_config={"config_list": autogen.config_list_from_json("OAI_CONFIG_LIST")}

)

user_proxy = RetrieveUserProxyAgent(
    name="rag_user_proxy",
    retrieve_config={
        "task": "qa",
        "docs_path": "data.txt"
    }
)

print("Welcome! You can start asking questions. Type 'exit' to quit.")


while True:
    user_query = input("You: ")
    
    if user_query.lower() == 'exit':
        break
    assistant.reset()
    # Initiate conversation with the user proxy agent and get the response
    response = user_proxy.initiate_chat(assistant, problem=user_query)
    
    print("Assistant:", response)
