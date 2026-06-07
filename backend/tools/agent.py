from typing import Sequence

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from backend.config import groq_chat_max_retries, groq_chat_model, require_groq_api_key


def create_agent(tools: Sequence[BaseTool]):
    """
    Groq + ReAct-style conversational agent with MCP tool access and chat memory.
    """
    llm = ChatGroq(
        model=groq_chat_model(),
        temperature=0,
        api_key=require_groq_api_key(),
        max_retries=groq_chat_max_retries(),
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return initialize_agent(
        tools=list(tools),
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
