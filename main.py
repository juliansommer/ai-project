from bs4 import BeautifulSoup
import chainlit as cl
from langchain import OpenAI, LLMMathChain
from langchain.agents import (
    create_csv_agent,
    initialize_agent,
    Tool,
)
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import (
    DuckDuckGoSearchRun,
    StructuredTool,
    WikipediaQueryRun,
)
from langchain.utilities import WikipediaAPIWrapper
import matplotlib
import os
import pandas as pd
import requests


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WIKIPEDIA_MAX_QUERY_LENGTH = 300
matplotlib.use("agg")


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {str(e)}")


def create_graph(csv_file_name: str, pandas_agent_query: str) -> str:
    """Creates an AI pandas agent to read the given csv file and create graphs using the data inside.
    Be specific about the data and the visualistion you want.
    The query should be a plain english description of the data you wish to retrieve and what to do with it.
    You do not need to ask the agent to read the CSV file, it is done automatically.
    You should always ask the agent to SAVE the created graph.
    The agent does not return true if it is complete so if there is no error assume the output file has been created"""
    try:
        pandas_agent = create_csv_agent(
                OpenAI(temperature=0),
                os.path.join(os.getcwd(), "ai_written_files", csv_file_name),
                verbose=True,
            )
    except:
        return "ERROR: Something went wrong when creating the CSV agent. Check the CSV file exists!"
    return pandas_agent.run(pandas_agent_query)


def wikipedia_table(link):
    res = requests.get(link)
    soup = BeautifulSoup(res.content,'lxml')
    table=soup.find_all('table',{'class':"wikitable"})
    df = pd.read_html(str(table))
    return(df) 


def write_to_file(file_name: str, file_content: str) -> bool:
    """Writes file_contents to a new file with the given file name.
    Ensure that the data being saved is only what is necessary to create the graph so only include the data points for each axis
    Returns a boolean indicating success (True) or failure (False)"""
    try:
        with open(os.path.join(os.getcwd(), "ai_written_files", file_name), "w", newline="", encoding="utf-8") as f:
            f.write(file_content)
    except:
        return False
    return True


@cl.on_chat_start
def start():
    create_folder_if_not_exists("ai_written_files")
    llm_math_chain = LLMMathChain.from_llm(OpenAI(temperature=0, streaming=True), verbose=True)
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math. Accepts a single string as input.",
        ),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions. Accepts a single string as input",
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful for gathering information about anything. data retrieved will also need to be trimmed. Accepts a single string as input.",
        ),
        Tool(
            name="Wikipedia Table",
            func=wikipedia_table,
            description="useful for specifically gathering tables from a wikipedia page. will need to call the function with the link to the wikipedia page you would like to get the table from. Accepts a single string as input.",
        ),
        StructuredTool.from_function(create_graph),
        StructuredTool.from_function(write_to_file),
    ]

    agent = initialize_agent(
        tools,
        ChatOpenAI(temperature=0, streaming=True, model="gpt-4"), 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    result = await cl.make_async(agent.run)(message, callbacks=[cb])
    await cl.Message(result).send()
