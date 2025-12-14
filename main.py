from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

from langchainhub import Client
hub = Client()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# Custom imports

from schemas import Source, AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS


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
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain101!")
    query = "Tell 3 AI/ML jobs that are posted on linked in recently?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    # Extract structured response
    response: AgentResponse = result["structured_response"]
    
    print("\n" + "="*50)
    print(f"INPUT: {query}")
    print("="*50)
    print(f"OUTPUT: {response.answer}")
    print("="*50)
    print("SOURCES:")
    for source in response.sources:
        print(f"  - {source.url}")
    print("="*50)


if __name__ == "__main__":
    main()
