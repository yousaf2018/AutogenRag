from flask import Flask, jsonify, request
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI
import os
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from flask_cors import CORS
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI"

genai.configure(api_key="AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI")
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})
# CORS(app, resources={r"/ask": {"origins": "https://bitpredict.ai"}})
# Load and preprocess your data
txt_file_path = 'data.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
data = text_splitter.split_documents(data)
# print(data)
# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(data, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Set up chat model
chat_model = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")


template = """Your name is Bitassist. Make sure your response to user very concise and conrete ,Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)
# while True:
#     question = input("Human input-->")
#     print(chain.invoke(question))


# messages = [
#     SystemMessage(
#         content="""
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
#         """
#     ),
#     HumanMessage(
#         content="Which is best model in term of total pnl"
#     ),
# ]
# message_1 = [
#     SystemMessage(
#         content="""
#                 You will get input user question, you will return 0 if user wants to know some generic information about bit predict platform or return 1 if user want to know about stratgies or models performance 
#         """
#     ),
#     HumanMessage(
#         content="Which is best model in term of total pnl"
#     ),
# ]
# while True:
#     question = str(input("Human input-->"))
#     resp = chat_model([
#     SystemMessage(
#         content="""
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
#         """
#     ),
#     HumanMessage(
#         content=question
#     ),
# ])
    # print(resp.content)
# Set up memory for conversation
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create conversation chain
# conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=chat_model,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#         )

@app.route('/ask', methods=['GET'])
def ask_question():
    if request.method == 'GET':
        # Extract question from the request query parameters
        question = request.args.get('question')

        if question:
            # Pass the question to the conversation chain
            result = chain.invoke(question)
            
            # Extract the answer from the result
            answer = str(result)

            # Return the answer as JSON response
            return jsonify({'response': answer}), 200
        else:
            return jsonify({'message': 'Question parameter is missing'}), 400
    else:
        return jsonify({'message': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True, port=5001)
