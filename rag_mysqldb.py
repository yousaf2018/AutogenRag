from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain.chat_models import ChatOpenAI
def generate(query: str) -> str:
    db_context = retrieve_from_db(query)
    
    system_message = """You are a professional representative of an Bitprdict and your name is Bitassist and respond only concise answers.
        You have to answer user's queries and provide relevant information to Bitpredict
        Example:
        
        Input:
        Which model is performing best in term of accuracy
        
        Context:
        You will query stats table and answer the question 
        
        
        """
    
    human_qry_template = HumanMessagePromptTemplate.from_template(
        """Input:
        {human_input}
        
        Context:
        {db_context}
        
        """
    )
    messages = [
      SystemMessage(content=system_message),
      human_qry_template.format(human_input=query, db_context=db_context)
    ]
    response = llm(messages).content
    return response

def retrieve_from_db(query: str) -> str:
    db_context = db_chain(query)
    db_context = db_context['result'].strip()
    return db_context

# Set up chat model
llm = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")



host = 'localhost'
port = str(3306)
username = 'root'
password = 'bitpredict'
database_schema = 'bitpredict'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"
# Connect to the database
engine = create_engine(mysql_uri)
# Test the connection
connection = engine.connect()
db = SQLDatabase.from_uri(mysql_uri, include_tables=['stats'])

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
while True:
    question = str(input("Human input-->"))
    # print(db_chain.invoke(question))
    retrieve_from_db(question)