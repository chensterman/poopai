from poopai.drivers import OpenAIDriver
from poopai.tools import GoogleSerperApiTool, WebScrapeTool
from poopai.agents import RAISEAgent, AgentResult

test_agent = RAISEAgent(
    driver=OpenAIDriver(model="gpt-4-turbo"),
    description="You are a web researcher that will find answers to my questions.",
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

for chunk in test_agent.execute("What's the difference between the ACT paper vs digital test?", stream=True):
    if isinstance(chunk, AgentResult):
        print("\nFINAL ANSWER:\n\n")
        print(chunk.content)
    else:
        print(chunk)
    print("\n---------------------------------------------\n")