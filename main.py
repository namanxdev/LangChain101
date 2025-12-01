from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily_search = TavilySearch()

# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches the web for information

#     Args:
#         query: The query to search for

#     Returns:
#         The search result
#     """
#     print(f"Searching the web for: {query}")
#     response = tavily_client.search(query)
#     return response


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [tavily_search]
agent = create_agent(
    model=llm,
    tools=tools,
)


def main():
    print("Hello from langchain101!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the latest news about india?")]})
    print(result)


if __name__ == "__main__":
    main()
