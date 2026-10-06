from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from collections.abc import Callable

from langgraph.graph import START, StateGraph, MessagesState  # pyright: ignore[reportMissingTypeStubs]
from langgraph.prebuilt import tools_condition, ToolNode

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools: list[Callable[[int, int], int | float]] = [add, multiply, divide]

# Define LLM with bound tools
llm = ChatAnthropic(
    model_name="claude-haiku-4-5-20251001",
    timeout=None,
    stop=None,
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)  # pyright: ignore[reportUnknownMemberType]

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState) -> MessagesState:
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()  # pyright: ignore[reportUnknownMemberType]
