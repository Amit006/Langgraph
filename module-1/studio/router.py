from langchain_anthropic import ChatAnthropic

from langgraph.graph import MessagesState  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import StateGraph, START, END  # pyright: ignore[reportMissingTypeStubs]
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
llm = ChatAnthropic(
    model_name="claude-haiku-4-5-20251001",
    timeout=None,
    stop=None,
    temperature=0,
)
llm_with_tools = llm.bind_tools([multiply]) # pyright: ignore[reportUnknownMemberType]

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm) # pyright: ignore[reportUnknownMemberType]
builder.add_node("tools", ToolNode([multiply]))  # pyright: ignore[reportUnknownMemberType]
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile() # pyright: ignore[reportUnknownMemberType]