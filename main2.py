from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


# Initialize components
tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Create the agent (latest LangChain way)
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain course\n")
    
    # Run the agent with a query
    result = agent.invoke({"messages": [("user", "What is weather in tokyo?")]})
    
    # Print the final response (handle different response formats)
    last_message = result["messages"][-1]
    if hasattr(last_message, 'content'):
        content = last_message.content
        # Handle list format (like from Gemini)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    print(item['text'])
                else:
                    print(item)
        else:
            print(content)
    else:
        print(last_message)


if __name__=="__main__":
    main()    