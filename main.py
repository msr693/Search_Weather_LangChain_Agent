import os
import certifi
from dotenv import load_dotenv
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langsmith import Client
from langchain.tools import tool
import requests





from langchain.agents import initialize_agent, Tool, create_react_agent, AgentExecutor



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("OPENAI_API_KEY:", OPENAI_API_KEY)
print("TAVILY_API_KEY:", TAVILY_API_KEY)

search_tool = TavilySearchResults(max_results=2)
result = search_tool.invoke("Give me latest news on AI?")
result

@tool
def get_weather_data(location: str) -> str:
    """
        Fetch current weather data for a city.
        """
    return f"The current weather in {location} is sunny with a temperature of 25°C."
    
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000)

client = Client()
prompt = client.pull_prompt("hwchase17/react")
prompt

tools = [search_tool,get_weather_data]
agent = create_react_agent(llm, tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke(
    {"input": "What is the capital of India?"
    "and then find its current weather."}
)
print("Final Output Start")
print("Response from LLM:", response)
print("Final Output End")