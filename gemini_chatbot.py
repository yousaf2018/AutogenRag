# !pip -q install langchain_experimental langchain_core
# !pip -q install google-generativeai==0.3.1
# !pip -q install google-ai-generativelanguage==0.4.0
# !pip -q install langchain-google-genai
# !pip -q install "langchain[docarray]"
# !pip show langchain langchain-core

#@title Setting up the Auth
import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap
from langchain_core.output_parsers import JsonOutputParser

# os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_AI_STUDIO2')
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI"

genai.configure(api_key="AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI")


model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = DocArrayInMemorySearch.from_texts(
    # mini docs for embedding
    [   "Is Bitpredict only focused on Bitcoin forecasting?\nWhile our primary focus is on Bitcoin forecasting, our platform is designed to adapt, and we have plans to expand our capabilities to cover other cryptocurrencies in the future.\n\n",
    "How often are new models added to Bitpredict?\nWe are continuously innovating and expanding our model offerings. New models are added regularly to ensure that users have access to the latest advancements in cryptocurrency forecasting.\n\n",
    "Is Bitpredict suitable for beginners in cryptocurrency trading?\nAbsolutely! Bitpredict is designed to cater to users of all experience levels. Our user-friendly interface, transparent reporting, and detailed backtest analyses provide valuable insights for both beginners and seasoned traders.\n\n",
    "Can I receive live forecasts directly through Bitpredict?\nWhile our forecasts are generally 10 minutes delayed on the website for free users, we are actively working on features to deliver live signals directly to your Slack channel, Telegram channel, or through email for subscribed models. Stay tuned for these upcoming enhancements.\n\n",
    "Will Bitpredict offer subscription plans in the future?\nYes, Bitpredict plans to introduce subscription plans in the near future. Our upcoming Pro Subscription will cater to users looking for advanced features such as real-time forecasts, pro API access, advanced analytics, preferred backtests, and other premium offerings.\n\n",
    "Models\n",
    "How are models at Bitpredict developed?\nWe leverage a multidisciplinary approach, integrating various fields such as machine learning, deep learning, and reinforcement learning to create robust and adaptive forecasting models followed by thorough backtesting, forwardtesting, and live simulation.\n\n",
    "Can I request a specific type of model or time horizon?\nWe value user feedback and suggestions. While we may not be able to fulfill every individual request, your input is crucial for our continuous improvement. Feel free to reach out through our contact channels with your suggestions.\n\n",
    "Are there any free models available on Bitpredict?\nYes, we offer free access to a selection of models. Users can explore and benefit from these models to understand our platforms capabilities before opting for premium features.\n\n",
    "How can I favorite models for later access?\nWe are actively working on introducing user management features, including the ability to favorite models for later access. Stay tuned for updates on this and more premium features.\n\n",
    "Why is there only Bitcoin forecasts?\nBitpredict was developed to provide a forward valuation for Bitcoin. However, the governing equations can be used for a variety of Proof of Work digital assets. In the future Bitpredict will release forecasts for a range of alternative cryptocurrencies.\n\n",
    "What time zone is used for forecasts?\nUTC, however all models reference your local machine clock.\n\n",
    "How have the ZT models been back tested?\nThe models have been thoroughly back tested on both seen and unseen data sets. The metrics shown on the website are a combination of the back tested results and the inclusion of live data moving forward. Every day new metrics are calculated based on the live performance of the previous day.\n\n",
    "Where do I find the accuracy statistics per model?\nThe accuracy of any model can be viewed directly at the bottom of its individual page. There you will find the specific drawdown, win/loss metrics, and general statistics that make up the overall performance of a model.\n\n",
    "What are the ranking criteria between model forecasts?\nAt the moment models are ranked based on their total achieved yield. Please note that this may not be the optimal metric to rank models.\n\n",
    "Performance\n",
    "How accurate are the forecasts on Bitpredict?\nForecast accuracy can vary based on market conditions, and it is important to note that cryptocurrency markets inherently involve risks. Our models undergo continuous refinement to enhance accuracy, and users can refer to backtest analyses for historical performance insights.\n\n",
    "Can I customize parameters for backtesting on Bitpredict?\nAbsolutely! Our Backtest page allows users to customize parameters such as take profit, stop loss, fee, and time stop. This customization empowers users to tailor backtests according to their trading preferences.\n\n",
    "What is the significance of the R2 score in the backtest analysis?\nThe R2 score measures the proportion of the variance in the dependent variable (actual PNL) that is predictable from the independent variables (forecasted PNL). A higher R2 score indicates a stronger relationship between forecasted and actual PNL.\n\n",
    "What is the significance of the Shape metric in backtest analysis?\nThe Shape metric, also known as the Sharpe ratio, assesses the risk-adjusted performance of a model. It provides insight into whether the returns of the model are due to a smart investment strategy or excessive risk. A higher Shape ratio indicates better risk-adjusted returns, making it a crucial metric for evaluating performance.\n\n",
    "How is the Sortino ratio important in assessing model performance?\nThe Sortino ratio, similar to the Sharpe ratio, measures risk-adjusted performance but focuses solely on the downside risk. It considers only the standard deviation of negative returns, providing a more insightful view of a models ability to withstand market downturns. A higher Sortino ratio suggests better downside risk management.\n\n",
    "What does the Alpha metric indicate in backtesting?\nAlpha represents the models excess return compared to its expected return, considering the inherent risks in the market. A positive alpha indicates the model has outperformed expectations, while a negative alpha suggests underperformance. It is a valuable metric for assessing a models ability to generate returns beyond what is predicted by its beta.\n\n",
    "How does the Beta metric contribute to backtest analysis?\nBeta measures a model sensitivity to market movements compared to a benchmark, often the overall market. A beta of 1 implies the model moves in sync with the market, while a beta greater than 1 indicates higher volatility. A beta less than 1 suggests lower volatility. Understanding beta is crucial for assessing how a model reacts to market fluctuations.\n\n",
    "What role do the 18 evaluation metrics play in backtest analysis?\nThe 18 evaluation metrics, including Shape, Sortino, Alpha, and Beta, provide a comprehensive view of a model performance. These metrics cover various aspects, such as drawdown, win/loss ratio, positive/negative PNL, and R2. Evaluating these metrics collectively allows users to gauge the model strengths and weaknesses across different dimensions, aiding in informed decision-making.\n\n",
    "What does long and short mean?\nIn futures trading lingo, long means one is buying a positive contract (if the price increases, one makes money from the appreciation and if the price decreases one losses money from the depreciation). The opposite of long is a short, where one is buying a negative contract (if the price decreases one makes money from the depreciation and if the price increases, one loses money from the appreciation).\n\n",
    "What time do the hypothetical trades open and close?\nPositions are typically opened and closed only if the model predicts a directional change.\n\n",
    "Why are there no risk controls (stop loss/take profit) being applied to the directional forecast of the models?\nOur goal is to prove that models resulting from the Bitpredict governing equations do produce superior foresight and performance without the need of risk control interventions. Risk controls themselves should only enhance a model’s already inherent performance to drive up yield consistency and reduce drawdown intensity and durations.\n\n",
    "Are trading fees taken into account within the performance analysis?\nNo. Trading fees are not taken into account as of yet. There will be a feature in the future that will allow users to observe the impact of trading fees for all individual models.\n\n",
    "How do we use the Bitpredict models to make money?\nWe do not provide financial or investment advice. Please treat the content of this entire website as an academic exercise to prove the existence of a robust pricing theory for Bitcoin.\n\n",
    "About\n",
    "How did Bitpredict come into existence?\nBitpredict was born out of a collective passion for unraveling the non-stationarity of Bitcoin prices. Our journey began with the realization that our team expertise could be harnessed to develop forward valuation models accessible to the public, showcasing the transformative power of AI in cryptocurrency forecasting.\n\n",
    "What motivated the development team at Bitpredict?\nThe main motivation was to bring years of expertise to use by decoding the complexities of cryptocurrency markets. We aimed to create forward valuation models that are not only advanced but also easily and readily accessible to the public, bridging the gap between intricate AI algorithms and user-friendly forecasting tools.\n\n",
    "What features can users expect in the near future from Bitpredict?\nIn the near future, Bitpredict plans to introduce features such as receiving signals directly into your Slack channel, Telegram channel, or through email for subscribed models. User management will also be introduced, offering premium features like real-time forecasts, pro API access, advanced analytics, preferred backtests, and more. Stay tuned for continuous updates and improvements.\n\n",
    "How can users stay informed about Bitpredict latest developments?\nUsers can stay updated on Bitpredict latest developments by following us on Twitter/X, joining our Telegram or Discord channels, and regularly checking our website for announcements. We also send out newsletters with important updates. For specific inquiries, users can reach out through our contact form.\n\n",
    "What is the main goal behind developing Bitpredict?\nThe primary goal was to utilize our team expertise to unravel the non-stationarity of cryptocurrency prices. By developing forward valuation models, we aimed to make AI-powered insights easily accessible to the public, demonstrating the capabilities of advanced forecasting tools in the realm of cryptocurrency.\n\n",
    ]
    ,
    embedding=embeddings 

)

retriever = vectorstore.as_retriever()

template = """Answer the question a full sentence, based only on the following context:
{context} 

Return you answer in three back ticks and answer does not found than reply with more generically

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

while True:
    question = str(input("Human input --->"))
    # print(chain.invoke({"question": question}))
    for s in chain.stream({"question": question}):
        print(s)