from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent

# Set up a search tool (here DuckDuckGo, but you can replace/add any!)
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for answering questions about current events or factual stuff"
    )
]

# 3. Initialize the agent
llm = OllamaLLM(model="mistral")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # zero-shot agent that uses tool descriptions
    verbose=True
)

# 4. Run the agent on a query
result = agent.run("Who won the FIFA World Cup in 2018 and what is Albert Einstein known for?")
print("\nAgent Output:", result)


