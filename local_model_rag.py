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
import google.generativeai as genai
from flask_cors import CORS

os.environ["GOOGLE_API_KEY"] = "AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI"

genai.configure(api_key="AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI")
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})
# CORS(app, resources={r"/ask": {"origins": "https://bitpredict.ai"}})
# Load and preprocess your data
txt_file_path = 'data.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(data, embedding=embeddings)

# Set up chat model
chat_model = ChatOpenAI(temperature=0.7, model_name="local-model", base_url="http://localhost:1234/v1", api_key="not-needed")

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create conversation chain
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
        )

@app.route('/ask', methods=['GET'])
def ask_question():
    if request.method == 'GET':
        # Extract question from the request query parameters
        question = request.args.get('question')

        if question:
            # Pass the question to the conversation chain
            result = conversation_chain({"question":question})
            
            # Extract the answer from the result
            answer = result["answer"]

            # Return the answer as JSON response
            return jsonify({'response': answer}), 200
        else:
            return jsonify({'message': 'Question parameter is missing'}), 400
    else:
        return jsonify({'message': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True, port=5001)
