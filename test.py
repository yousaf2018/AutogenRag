# from sqlalchemy import create_engine
# from langchain.prompts.chat import ChatPromptTemplate
# from langchain.agents import AgentType, create_sql_agent
# from langchain.sql_database import SQLDatabase
# from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain.chat_models import ChatOpenAI

# db_engine = create_engine("mysql+pymysql://root:bitpredict@localhost/bitpredict")

# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", 
#          """
#     "You are an expert in converting English questions to SQL code! The SQL database has the name Stats and has the following columns - rank, strategy_name, current_drawdown, curr_drawdown_duration, average_drawdown, average_drawdown_duration, max_drawdown, max_drawdown_duration, r2_score, sharpe, sortino, total_pnl, average_daily_pnl, win_loss_ratio, total_positive_pnl, total_negative_pnl, total_wins, total_losses, consective_wins, consective_losses, win_percentage, loss_percentage, pnl_sum_1, pnl_sum_7, pnl_sum_15, pnl_sum_30, pnl_sum_45, pnl_sum_60, alpha, beta.\n\n",
#     "For example,\n",
#     "Example 1 - How many entries have a rank present?\n",
#     "SELECT COUNT(*) FROM Stats;\n\n",
#     "Example 2 - How many entries have a current_drawdown greater than 0.5?\n",
#     "SELECT COUNT(*) FROM Stats WHERE current_drawdown > 0.5;\n\n",
#     "Example 3 - What is the average drawdown?\n",
#     "SELECT AVG(average_drawdown) FROM Stats;\n\n",
#     "Example 4 - What is the maximum drawdown duration?\n",
#     "SELECT MAX(max_drawdown_duration) FROM Stats;\n\n",
#     "Don't include ``` and \\n in the output",         
#     """
#          ),
#         ("user", "{question}\n ai: "),
#     ]
# )
# llm = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")

# db = SQLDatabase(db_engine)

# sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# sql_toolkit.get_tools()

# sqldb_agent = create_sql_agent(
#     llm=llm,
#     toolkit=sql_toolkit,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# while True:
#     question = input(str("Human input -->"))
#     sqldb_agent.run(final_prompt.format(
#             question="Which strategy is best in term of total pnl"
#     ))




from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI"

cs="mysql+pymysql://root:bitpredict@localhost/bitpredict"
db_engine=create_engine(cs)
db=SQLDatabase(db_engine)

safety_settings=[
  {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]


genai.configure(api_key="AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI")

llm = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")
                             
sql_toolkit=SQLDatabaseToolkit(db=db,llm=llm)
sql_toolkit.get_tools()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user and converting into sql queriesa to generate correcrt answer.
        Please use the below context to write the microsoft sql queries , dont use mysql queries.
       context:
       you must query against the connected database, The SQL database has the name Stats and has the following columns - rank, strategy_name, current_drawdown, curr_drawdown_duration, average_drawdown, average_drawdown_duration, max_drawdown, max_drawdown_duration, r2_score, sharpe, sortino, total_pnl, average_daily_pnl, win_loss_ratio, total_positive_pnl, total_negative_pnl, total_wins, total_losses, consective_wins, consective_losses, win_percentage, loss_percentage, pnl_sum_1, pnl_sum_7, pnl_sum_15, pnl_sum_30, pnl_sum_45, pnl_sum_60, alpha, beta.\n\n",
    "For example,\n",
    "Example 1 - How many entries have a rank present?\n",
    "SELECT COUNT(*) FROM Stats;\n\n",
    "Example 2 - How many entries have a current_drawdown greater than 0.5?\n",
    "SELECT COUNT(*) FROM Stats WHERE current_drawdown > 0.5;\n\n",
    "Example 3 - What is the average drawdown?\n",
    "SELECT AVG(average_drawdown) FROM Stats;\n\n",
    "Example 4 - What is the maximum drawdown duration?\n",
    "SELECT MAX(max_drawdown_duration) FROM Stats;\n\n",
    "Don't include ``` and \\n in the output",
    
        """
        ),
        ("user","{question}\ ai: ")
    ]
)
agent=create_sql_agent(llm=llm,toolkit=sql_toolkit,verbose=True, handle_parsing_errors=True)
# result = agent.run(prompt.format_prompt(question="which strategy performed best in term sortino"))
print(agent.invoke("Print all sql tables names only"))