"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.runtime import get_runtime


from react_agent.context import Context


# Test version with 2 dummy docs about LangChain and LangGraph
from react_agent.langchain_doc_retriever import langchain_doc_retriever
# Version that actually loads LangChain documentation
from react_agent.langsmith_doc_retriever_complete import get_langsmith_doc_retriever

# Uncomment to ingest actual LangChain doc,
# it will take some time to run on the first iteration (in minutes)
# result = langsmith_doc_retriever.invoke(query)
langsmith_doc_retriever = get_langsmith_doc_retriever()


async def search(query: str) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearchResults(
        max_results=runtime.context.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


@tool
def search_langchain(
    query: str
) -> Optional[list[dict[str, Any]]]:
    """Search in LangChain and LangGraph official documentation.

    This tool should be used in priority over a generic web search. 

    It is is designed to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about LangChain and LangGraph.
    """
    # result = retriever.invoke(query)
    result = langchain_doc_retriever.invoke(query)
    return cast(list[dict[str, Any]], result)


print(search_langchain.description)


def search_langsmith(
    query: str
) -> Optional[list[dict[str, Any]]]:
    """Search in LangSmith official documentation.

    This tool should be used in priority over a generic web search. 

    It is is designed to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about LangSmith
    """
    # result = retriever.invoke(query)
    result = langsmith_doc_retriever.invoke(query)
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search, search_langsmith, search_langchain]
