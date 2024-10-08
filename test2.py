# Install necessary packages
# !pip -q install langchain_experimental langchain_core
# !pip -q install google-generativeai==0.3.1
# !pip -q install google-ai-generativelanguage==0.4.0
# !pip -q install langchain-google-genai
# !pip -q install "langchain[docarray]"
# !pip install psycopg2-binary
# !pip show langchain langchain-core

# Import necessary libraries
import os
import psycopg2
from psycopg2 import OperationalError
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap
from langchain_core.output_parsers import JsonOutputParser
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

# Database connection parameters
db_params = {
    'host': '193.22.147.204',
    'port': '5432',
    'dbname': 'bitpredict',
    'user': 'admin',
    'password': '9Hd2mTg5Kw'
}

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(**db_params)
        print("Database successfully connected")
        return connection
    except OperationalError as e:
        print(f"Database connection error: {e}")
        return None

# Initialize the database connection
connection = get_db_connection()
cursor = connection.cursor()

# Safety settings for the model
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI"

genai.configure(api_key="AIzaSyB8eLanCYOHzsW-BVexCV1T7uKMeLRsTUI")

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

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
    "The information provided on this website does not constitute investment advice, financial advice, trading advice, or any other sort of advice and you should not treat any of this website's content as such. Zero Theorem Pty Ltd does not recommend that any cryptocurrency should be bought, sold, or held by you. Do conduct your own due diligence and consult your financial advisor before making any investment decisions.",
    "Our Mission",
    "Decoding Dynamics",
    "At Bitpredict, our mission transcends the prediction of Bitcoin prices; we aim to decode the dynamic intricacies of cryptocurrency markets. Fueled by the relentless pursuit of knowledge, our seasoned developers, armed with high doctoral degrees and years of practical experience, strive to unravel the multifaceted layers that shape the crypto landscape. In an ever-changing environment, our commitment to decoding dynamics extends beyond algorithms. Understanding market sentiments, macroeconomic factors, and emerging trends. Each model crafted is our dedication to not only forecast but also comprehend the fundamental forces driving Bitcoin's trajectory.",
    "Transparent Reporting",
    "Transparency is the bedrock upon which Bitpredict stands. In our mission to provide users with unparalleled insights, we go beyond traditional backtesting; we offer transparent reporting. The heartbeat of our models pulsates on our Backtest page, where users witness live analyses of Profit and Loss (PNL), drawdowns, and 18 evaluation metrics. Our commitment to transparent reporting ensures that users don't merely see predictions but gain a clear, unobstructed view into the performance metrics, fostering informed decision-making. In a world where trust is paramount, Bitpredict pioneers a new standard with reporting that transcends mere predictions.",
    "Unparalleled Accessibility",
    "Our mission extends beyond decoding dynamics and transparent reporting to making forward valuation models effortlessly accessible. The user-friendly home page features a filterable and sortable grid, providing a visual gateway for enthusiasts to seamlessly explore and access models based on diverse time horizons and an array of performance metrics. Accessibility is not just about convenience; it's about empowering users with the tools they need to navigate the intricate cryptocurrency landscape. Bitpredict ensures that advanced forecasting is not confined to experts but is accessible to anyone with a curiosity about the future of Bitcoin.",
    "Our Expertise",
    "In the ever-evolving landscape of cryptocurrency, possessing state-of-the-art expertise is paramount. At Bitpredict, we recognize that success in decoding the intricacies of Bitcoin pricing demands not only advanced algorithms but also a deep understanding of market dynamics. Our team, comprising seasoned developers with high doctoral degrees and years of hands-on experience, is at the forefront of this endeavor.",
    "Cryptocurrency markets are notorious for their volatility, making them a fascinating yet challenging domain to navigate. It is in this volatile environment that our team's extensive academic background and practical experience come to the fore. Armed with doctoral-level degrees, we bring a wealth of theoretical knowledge, combined with the acumen developed through years of research and development. This combination of academic rigor and real-world application positions us uniquely in the realm of cryptocurrency forecasting.",
    "Our commitment to staying ahead in this rapidly changing landscape is reflected in the continuous refinement and expansion of our models. As believers in the power of expertise, we understand that the cryptocurrency market is not just about numbers and algorithms but also about interpreting market sentiment, understanding macroeconomic factors, and adapting to unforeseen events.",
    "The depth of our expertise is intricately woven into the fabric of Bitpredict. It's not merely about predicting price movements but comprehending the underlying factors that drive them. Each model we create is a manifestation of our collective expertise, finely tuned to not only forecast but also provide users with insights that go beyond the surface.",
    "In the pursuit of accurate forecasting, we employ a multidisciplinary approach, integrating various fields such as machine learning, deep learning, and reinforcement learning. This diverse skill set, combined with our doctoral-level education, enables us to develop models that not only adapt to market changes but also anticipate them.",
    "In essence, our expertise is not just a tool; it's a compass that guides us through the complexities of the cryptocurrency market. It's a testament to our dedication to delivering not just predictions but informed and insightful analyses. Bitpredict stands as a beacon of expertise in a landscape where knowledge is the currency that truly holds value.",
    "Our Manifesto",
    "Even though we use a combination of statistical, econometric, and machine learning methods to estimate value, we will never overlook the stochastic properties of reality for the pristine beauty of mathematical formulation. We present clear assumptions, oversights, and accuracy of our models to those who intend to use them and never claim that they have the full explanatory insights into real-world market dynamics. We understand that our models can provoke past, current, and future economic theories and accept that our work may result in consequences on society that can be far beyond our own comprehension. We commit to continuously improving and evolving our models and will never claim that they are complete in explanatory power. We dedicate ourselves to expanding the current body of knowledge by releasing our own research to the public. We specifically express that our models or their outputs are not intended to be used as financial advice.",
    "Our Why",
    "“In the vast and dynamic realm of financial markets, the why is often the driving force behind innovation. Bitpredict was born out of a collective passion for unraveling the complexities of the cryptocurrency market. As a team of seasoned developers with high doctoral degrees and years of experience in research and development, we recognized the need for sophisticated tools to navigate the ever-changing landscape of digital assets. Our journey began with a simple question: How can we harness the power of artificial intelligence to decode the non-stationarity of Bitcoin prices? This question fueled late-night brainstorming sessions, rigorous research, and the relentless pursuit of a solution that could democratize advanced forecasting models. The motivation was clear – to make forward valuation accessible, transparent, and insightful for everyone. Over time, Bitpredict has evolved into a dynamic hub with over 1200 models catering to various time horizons. Our commitment to transparency led us to provide live backtests for every model, enabling users to witness real-time performance and make informed decisions. The filterable grid on our home page reflects our dedication to user-friendly accessibility, allowing enthusiasts to explore models based on their specific criteria. Our manifesto is not just a set of principles; it's a living, breathing commitment to our users. It's a promise to continually refine and expand our models, adapting to the ever-evolving needs of the cryptocurrency community. The journey from inception to today has been marked by continuous learning, innovation, and a relentless pursuit of excellence. Join us on this journey. Explore the depths of Bitpredict, where knowledge meets innovation, and decoding the future of Bitcoin is not just a mission but a shared passion. Contact us through our various channels – Twitter/X, Telegram, Discord, or email at info@bitpredict.ai – and become a part of the ongoing narrative at Bitpredict. Stay tuned for exciting features, including real-time forecasts, pro API access, and user management options. The journey is vibrant, and the best is yet to come."


    ]
    ,
    embedding=embeddings 

)
retriever = vectorstore.as_retriever(search_type="mmr")

template_query = """
You are sql expert. You will generate sql query from user question by using below database schema and make sure donot fetch all columns and rows at once and fetch it 
smartly like top 10 etc 
Database Schema:
CREATE TABLE "stats"."models_stats"(
   "frontend_model_name" character varying,
   "time_horizon" character varying,
   "symbol" character varying,
   "rolling_window" character varying,
   "start_date" bigint,
   "last_forecast" bigint,
   "next_forecast" bigint,
   "best_performing_conditions" character varying,
   "pnl_percent" double precision,
   "current_prediction" character varying,
   "entry_price" double precision,
   "current_price" double precision,
   "current_pnl" double precision,
   "avg_daily_pnl" double precision,
   "pnl_1d" double precision,
   "pnl_7d" double precision,
   "pnl_15d" double precision,
   "pnl_30d" double precision,
   "pnl_45d" double precision,
   "pnl_60d" double precision,
   "total_long_pnl" double precision,
   "avg_long_pnl_per_trade" double precision,
   "num_long_trades" double precision,
   "win_rate_long_trades" double precision,
   "avg_long_trade_duration" double precision,
   "max_long_trade_pnl" double precision,
   "min_long_trade_pnl" double precision,
   "pct_long_trades" double precision,
   "total_short_pnl" double precision,
   "avg_short_pnl_per_trade" double precision,
   "num_short_trades" double precision,
   "win_rate_short_trades" double precision,
   "avg_short_trade_duration" double precision,
   "max_short_trade_pnl" double precision,
   "min_short_trade_pnl" double precision,
   "pct_short_trades" double precision,
   "total_return" double precision,
   "cagr" double precision,
   "monthly_return" double precision,
   "weekly_return" double precision,
   "daily_return" double precision,
   "sharpe_ratio" double precision,
   "sortino_ratio" double precision,
   "calmar_ratio" double precision,
   "alpha" double precision,
   "beta" double precision,
   "r2" double precision,
   "information_ratio" double precision,
   "treynor_ratio" double precision,
   "profit_factor" double precision,
   "omega_ratio" double precision,
   "gain_to_pain_ratio" double precision,
   "max_drawdown" double precision,
   "max_drawdown_days" double precision,
   "avg_drawdown" double precision,
   "avg_drawdown_days" double precision,
   "drawdown_duration" double precision,
   "current_drawdown" double precision,
   "current_drawdown_days" double precision,
   "var_95" double precision,
   "cvar_95" double precision,
   "volatility" double precision,
   "downside_deviation" double precision,
   "tail_ratio" double precision,
   "skewness" double precision,
   "kurtosis" double precision,
   "number_of_trades" double precision,
   "win_rate" double precision,
   "loss_rate" double precision,
   "average_win" double precision,
   "average_loss" double precision,
   "average_trade_duration" double precision,
   "largest_win" double precision,
   "largest_loss" double precision,
   "consecutive_wins" double precision,
   "consecutive_losses" double precision,
   "avg_trade_return" double precision,
   "profitability_per_trade" double precision,
   "total_profit" double precision,
   "total_loss" double precision,
   "net_profit" double precision,
   "gross_profit" double precision,
   "gross_loss" double precision,
   "avg_profit_per_trade" double precision,
   "avg_loss_per_trade" double precision,
   "profit_loss_ratio" double precision,
   "winning_months" double precision,
   "losing_months" double precision,
   "winning_weeks" double precision,
   "losing_weeks" double precision,
   "percentage_positive_months" double precision,
   "percentage_negative_months" double precision,
);

CREATE UNIQUE INDEX models_stats_pkey ON stats.models_stats USING btree (frontend_model_name);



For example,
Example 1 - How many entries have a rank present?
SELECT COUNT(*) FROM stats.models_stats WHERE rank IS NOT NULL;

Example 2 - How many entries have a current_drawdown greater than 0.5?
SELECT COUNT(*) FROM stats.models_stats WHERE current_drawdown > 0.5;

Example 3 - What is the average drawdown?
SELECT AVG(avg_drawdown) FROM stats.models_stats;

Example 4 - What is the maximum drawdown duration?
SELECT MAX(max_drawdown_days) FROM stats.models_stats;

When you are replying with an SQL query, reply with backticks like this ```SELECT * FROM stats.models_stats``` and make sure only one sql query for each question. Make sure query should be correct with nor error
and do not make it complex query please 
Question: {question}
"""


template_generic_qa = """
Your name is Bitassit to assist users queries related Bitpredict. You should tell that you are made by Bitpredict team.
Reply generic according to query and donot tell i have no context for question and just reply with normal answer by your understanding. Make sure tell with message I have only information related to Bitpredict 

Try to chit chat with users professionally and answer user quries according to given context if something matches in context 
{context} 
Question: {question}

Reply generic according to query and donot tell i have no context for question and just reply with normal answer by your understanding
"""


template_result_explain_query = """
Your task to explain to user in a conscise way result of database query and make sure donot tell user about query and database 
Here is query 
{query} 
Here is result of query you will explain to user in a simple way 
Result: {result}
"""
template_classification = """
You will reply only 1 if user question requried database connection and query. Otherwise reply 0 if user asks generic question. You must reply only 1 or 0


Here is database schema for better understanding user question 
Database Schema:
CREATE TABLE "stats"."models_stats"(
   "backend_model_name" character varying NOT NULL,
   "frontend_model_name" character varying,
   "time_horizon" character varying,
   "symbol" character varying,
   "rolling_window" character varying,
   "start_date" bigint,
   "last_forecast" bigint,
   "next_forecast" bigint,
   "best_performing_conditions" character varying,
   "pnl_percent" double precision,
   "current_prediction" character varying,
   "entry_price" double precision,
   "current_price" double precision,
   "current_pnl" double precision,
   "avg_daily_pnl" double precision,
   "pnl_1d" double precision,
   "pnl_7d" double precision,
   "pnl_15d" double precision,
   "pnl_30d" double precision,
   "pnl_45d" double precision,
   "pnl_60d" double precision,
   "total_long_pnl" double precision,
   "avg_long_pnl_per_trade" double precision,
   "num_long_trades" double precision,
   "win_rate_long_trades" double precision,
   "avg_long_trade_duration" double precision,
   "max_long_trade_pnl" double precision,
   "min_long_trade_pnl" double precision,
   "pct_long_trades" double precision,
   "total_short_pnl" double precision,
   "avg_short_pnl_per_trade" double precision,
   "num_short_trades" double precision,
   "win_rate_short_trades" double precision,
   "avg_short_trade_duration" double precision,
   "max_short_trade_pnl" double precision,
   "min_short_trade_pnl" double precision,
   "pct_short_trades" double precision,
   "total_return" double precision,
   "cagr" double precision,
   "monthly_return" double precision,
   "weekly_return" double precision,
   "daily_return" double precision,
   "sharpe_ratio" double precision,
   "sortino_ratio" double precision,
   "calmar_ratio" double precision,
   "alpha" double precision,
   "beta" double precision,
   "r2" double precision,
   "information_ratio" double precision,
   "treynor_ratio" double precision,
   "profit_factor" double precision,
   "omega_ratio" double precision,
   "gain_to_pain_ratio" double precision,
   "max_drawdown" double precision,
   "max_drawdown_days" double precision,
   "avg_drawdown" double precision,
   "avg_drawdown_days" double precision,
   "drawdown_duration" double precision,
   "current_drawdown" double precision,
   "current_drawdown_days" double precision,
   "var_95" double precision,
   "cvar_95" double precision,
   "volatility" double precision,
   "downside_deviation" double precision,
   "tail_ratio" double precision,
   "skewness" double precision,
   "kurtosis" double precision,
   "number_of_trades" double precision,
   "win_rate" double precision,
   "loss_rate" double precision,
   "average_win" double precision,
   "average_loss" double precision,
   "average_trade_duration" double precision,
   "largest_win" double precision,
   "largest_loss" double precision,
   "consecutive_wins" double precision,
   "consecutive_losses" double precision,
   "avg_trade_return" double precision,
   "profitability_per_trade" double precision,
   "total_profit" double precision,
   "total_loss" double precision,
   "net_profit" double precision,
   "gross_profit" double precision,
   "gross_loss" double precision,
   "avg_profit_per_trade" double precision,
   "avg_loss_per_trade" double precision,
   "profit_loss_ratio" double precision,
   "winning_months" double precision,
   "losing_months" double precision,
   "winning_weeks" double precision,
   "losing_weeks" double precision,
   "percentage_positive_months" double precision,
   "percentage_negative_months" double precision,
   "downsampled_pnl_sum" json
);

CREATE UNIQUE INDEX models_stats_pkey ON stats.models_stats USING btree (backend_model_name);



For example,
Example 1 - How many entries have a rank present?
SELECT COUNT(*) FROM stats.models_stats WHERE rank IS NOT NULL;

Example 2 - How many entries have a current_drawdown greater than 0.5?
SELECT COUNT(*) FROM stats.models_stats WHERE current_drawdown > 0.5;

Example 3 - What is the average drawdown?
SELECT AVG(avg_drawdown) FROM stats.models_stats;

Example 4 - What is the maximum drawdown duration?
SELECT MAX(max_drawdown_days) FROM stats.models_stats;

You will only reply 1 or 0 according user question if required database connection and query than reply with 1 or other 0 for generic question 
{context} 
Question: {question}
"""
prompt1 = ChatPromptTemplate.from_template(template_query)
prompt2 = ChatPromptTemplate.from_template(template_classification)
prompt3 = ChatPromptTemplate.from_template(template_generic_qa)
prompt4 = ChatPromptTemplate.from_template(template_result_explain_query)

output_parser = StrOutputParser()

query_chain = RunnableMap({
    "question": lambda x: x["question"]
}) | prompt1 | model | output_parser


classification_chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt2 | model | output_parser

generic_qa_chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt3 | model | output_parser


query_result_explain_chain = RunnableMap({
    "result": lambda x: x["result"],
    "query": lambda x: x["query"]
}) | prompt4 | model | output_parser


@app.route('/ask', methods=['GET'])
def ask_question():
    global connection, cursor
    if request.method == 'GET':
        question = request.args.get('question')

        if question:
            try:
                ans_text = ""
                for answer in classification_chain.stream({"question": question}):
                    ans_text += answer

                if int(ans_text) == 0:
                    ans_text = ""
                    for answer in generic_qa_chain.stream({"question": question}):
                        ans_text += answer
                    return jsonify({'response': ans_text}), 200

                elif int(ans_text) == 1:
                    ans_text = ""
                    for answer in query_chain.stream({"question": question}):
                        ans_text += answer

                    if "sql" in ans_text or "```" in ans_text or "SELECT" in ans_text:
                        # Clean the SQL query text
                        ans_text = ans_text.replace("```", "").replace("sql", "").strip()

                        # Try executing the query, and attempt reconnection on failure
                        for _ in range(2):  # Retry up to 2 times
                            try:
                                if not connection:
                                    connection = get_db_connection()
                                    cursor = connection.cursor()

                                # Execute the cleaned SQL query
                                cursor.execute(ans_text)
                                rows = cursor.fetchall()
                                explained_result = ""
                                for answer in query_result_explain_chain.stream({"result": rows, "query": ans_text}):
                                    explained_result += answer

                                return jsonify({'response': explained_result}), 200

                            except Exception as db_exception:
                                # Handle database connection or cursor issues
                                connection = get_db_connection()
                                cursor = connection.cursor()
                        # If it still fails, raise the exception to the outer handler
                        raise db_exception

                    else:
                        ans_text = ""
                        for answer in generic_qa_chain.stream({"question": question}):
                            ans_text += answer
                        return jsonify({'response': ans_text}), 200

                else:
                    ans_text = ""
                    for answer in generic_qa_chain.stream({"question": question}):
                        ans_text += answer
                    return jsonify({'response': ans_text}), 200

            except Exception as e:
                return jsonify({'response': f'An error occurred while processing your request: {str(e)}'}), 500
        else:
            return jsonify({'message': 'Question parameter is missing'}), 400
    else:
        return jsonify({'message': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True,port=5001)

