from dotenv import load_dotenv
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute Python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of tour code to answer questions.
    You might know th answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 2 QR codes that point to www.google.com,
    #         you have qrcode package installed already."""
    #     }
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    # csv_agent.invoke(
    #    input={
    #        "input": "how many columns are there in the file episode_info.csv?"
    #    }
    # )

    csv_agent.invoke(
        input={
            "input": "how many episodes did Larry David write?"
        }
    )


if __name__ == "__main__":
    main()
