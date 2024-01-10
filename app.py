import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

llm_config = {"config_list": autogen.config_list_from_json("OAI_CONFIG_LIST")}

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpfull assistant",
    llm_config=llm_config
)

user_proxy = RetrieveUserProxyAgent(
    name="rag_user_proxy",
    retrieve_config={
        "task": "qa",
        "docs_path": "data.txt"
    }
)

assistant.reset()
user_proxy.initiate_chat(assistant, problem="In what timeframe can I return an item?")