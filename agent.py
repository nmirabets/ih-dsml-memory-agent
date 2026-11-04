import streamlit as st
from openai import OpenAI
import json
from tavily import TavilyClient
from memory import save_memory

MODEL = "gpt-6-luna"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# TOOLS
def search_web(query):
    # Initialize Tavily
    tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
    return "\n".join(f"{{Content: {result['content']} \n Source: {result['url']}}}" for result in tavily.search(query, search_depth="basic")["results"])


FUNCTIONS = {"save_memory": save_memory, "search_web": search_web}

TOOLS = [
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "Save a memories to the database.",
        "parameters": {"type": "object",
                       "properties": {"memories": {"type": "array", "items": {"type": "string"}}},
                       "required": ["memories"], "additionalProperties": False},
        "strict": True}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for information.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
        "strict": True}},
]

def run_agent(messages, max_steps=5):
    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, reasoning_effort="none")
        message = response.choices[0].message
        if not message.tool_calls:                       # no more tools: this is the answer
            return message.content
        messages.append(message)
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments)
            result = FUNCTIONS[call.function.name](**arguments)
            print(f"step {step}: {call.function.name}({arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return "Stopped: too many steps."