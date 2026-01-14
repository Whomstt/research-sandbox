# References: https://docs.langchain.com/oss/python/langgraph/quickstart#use-the-functional-api

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolCall
from langgraph.func import entrypoint, task

from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> int:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


@task
def call_llm(messages: list[BaseMessage]):
    return llm_with_tools.invoke(
        [
            SystemMessage(
                content="You are a great mathematician that will help me with maths"
            )
        ]
        + messages
    )


@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


@entrypoint()
def agent(messages: list[BaseMessage]):
    llm_response = call_llm(messages).result()
    while True:
        if not llm_response.tool_calls:
            break
        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_llm(messages).result()
    messages = add_messages(messages, llm_response)
    return messages


# Invoke
messages = [HumanMessage(content=input("Enter a mathematical question: "))]
for chunk in agent.stream(messages, stream_node="updates"):
    print(chunk)
    print("\n")
