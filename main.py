from dotenv import load_dotenv
import json
import re

load_dotenv()

from langchain.agents import create_agent as create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# Custom imports
from schemas import Source, AgentResponse

# Initialize tools and LLM
tavily_search = TavilySearch(max_results=5)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [tavily_search]

# Create the react agent using langgraph (modern approach)
agent = create_react_agent(model=llm, tools=tools)


def extract_sources_from_messages(messages) -> list[Source]:
    """Extract source URLs from agent tool messages."""
    sources = []
    seen_urls = set()
    
    for msg in messages:
        # Check for tool messages with Tavily results
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Try to parse JSON content from tool responses
            try:
                if 'url' in msg.content.lower():
                    # Find URLs in the content
                    url_pattern = r'https?://[^\s\'"<>)}\]]+' 
                    urls = re.findall(url_pattern, msg.content)
                    for url in urls:
                        # Clean up URL
                        url = url.rstrip('.,;:')
                        if url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(Source(url=url))
            except Exception:
                pass
    
    return sources


def run_agent(query: str) -> AgentResponse:
    """Run the agent and return structured response."""
    # Invoke agent with messages format
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract the final answer from the last AI message
    messages = result.get("messages", [])
    answer = ""
    
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.type == "ai":
            content = msg.content
            # Handle content that might be a list (multimodal format)
            if isinstance(content, list):
                # Extract text from list of content blocks
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                answer = '\n'.join(text_parts)
            elif isinstance(content, str):
                answer = content
            
            if answer:
                break
    
    # Extract sources from tool messages
    sources = extract_sources_from_messages(messages)
    
    return AgentResponse(answer=answer, sources=sources)

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