from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}!"


agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in pileru india"}]},stream=True
)
print(res["messages"][-1].content[0]["text"])