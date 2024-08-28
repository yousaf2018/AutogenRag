from langchain_community.utilities import SQLDatabase
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI


host = 'localhost'
port = str(3306)
username = 'root'
password = 'bitpredict'
database_schema = 'bitpredict'
mysql_uri = "mysql+pymysql://root:bitpredict@localhost/bitpredict"
# Connect to the database
# engine = create_engine(mysql_uri)
# Test the connection
# connection = engine.connect()
db = SQLDatabase.from_uri(mysql_uri)
# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Stats LIMIT 10;")

llm = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")

class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(['stats', 'strategies'])
system = f"""Answer user question from given table names using sql quries and return result of query also
{table_names}
"""
table_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
# Assuming your input query is correctly formatted
input_query = "Which model performed best in term of total pnl"

# Create the input dictionary with the required 'input' key
input_data = {"input": input_query}

# Invoke the chain with the input data
query = table_chain.invoke(input_data)

# Print or handle the query response
print(query)