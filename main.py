from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# Custom imports
from schemas import AgentResponse

# Initialize tools and LLM
tavily_search = TavilySearch(max_results=5)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [tavily_search]

# Create the react agent
agent = create_agent(model=llm, tools=tools)

# Create a structured output LLM for formatting the final response
structured_llm = llm.with_structured_output(AgentResponse)


def run_agent(query: str) -> AgentResponse:
    """Run the agent and return structured response."""
    # Invoke agent with messages format
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract the final answer from the last AI message
    messages = result.get("messages", [])
    raw_answer = ""
    tool_results = []
    
    for msg in messages:
        # Collect tool results for context
        if msg.type == "tool" and hasattr(msg, 'content'):
            tool_results.append(msg.content)
        
        # Get the final AI answer
        if msg.type == "ai" and hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                raw_answer = '\n'.join(text_parts)
            elif isinstance(content, str):
                raw_answer = content
    
    # Use structured output to format the response with sources
    formatting_prompt = f"""Based on this agent response and tool results, create a structured response.

Agent Answer: {raw_answer}

Tool Results (contains source URLs): {tool_results[:2] if tool_results else 'None'}

Extract the answer and any source URLs from the tool results."""

    structured_response: AgentResponse = structured_llm.invoke(formatting_prompt)
    
    return structured_response

def main():
    print("Hello from langchain101!")
    query = "Tell 3 AI/ML jobs that are posted on LinkedIn recently?"
    
    print(f"\n{'='*50}")
    print(f"INPUT: {query}")
    print("="*50)
    print("Running agent...\n")
    
    try:
        result: AgentResponse = run_agent(query)
        
        print("\n" + "="*50)
        print("OUTPUT:")
        print(result.answer)
        print("="*50)
        print("SOURCES:")
        if result.sources:
            for source in result.sources:
                print(f"  - {source.url}")
        else:
            print("  - No sources found.")
        print("="*50)
        
    except Exception as e:
        print(f"Agent failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()