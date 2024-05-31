import sys
sys.path.append('..')
from poopai.drivers import OpenAIDriver
from poopai.tools import GoogleSerperApiTool, WebScrapeTool
from poopai.agents import RAISEAgent, AgentResult


description = """
You are a financial news analysis AI tasked with synthesizing the latest news relevant to a specific investment portfolio.
"""

task = """
**Portfolio:** [Tesla, NVIDIA]

**Risks to Monitor:** [Supply chain, China, interest rates]

**Ignore:** [Elon Musk’s compensation package]

**Instructions:**
1. Create an ongoing list of all google searches you've made. Return each time.
1. Scan financial news sources for relevant articles, focusing on the listed risks.
2. Evaluate newsworthiness by considering immediate impact, risk magnitude, and long-term effects. Summarize each article in max 15 words with the source (name only) and date.
3. Finally, create a daily summary report with the most newsworthy articles, sorted by date.
```
"""


task2 = """
**Portfolio:** [Tesla, NVIDIA]

**Risks to Monitor:** [Supply chain, China, interest rates]

**Ignore:** [Elon Musk’s compensation package]

**Instructions:**
1. First, create a list of all searches we will make.
2. Next, go through each search and find betwween zero and 2 relevant articles, based on newsworthiness by considering immediate impact.
3. Summarize each article in max 15 words with the source (name only) and date.
3. Finally, create a daily summary report with the most newsworthy articles, sorted by date, which contains all accumulated snippets.
4. Call agent_finish when you are ready. Never repeat a Google Search.
```
"""



task3 = """
**Portfolio:** [Tesla, NVIDIA]

**Risks to Monitor:** [Supply chain, China, interest rates]

**Ignore:** [Elon Musk’s compensation package]

"""


agent = RAISEAgent(
    driver=OpenAIDriver(model="gpt-4o"),
    description=description,
    tools=[GoogleSerperApiTool(), WebScrapeTool()],
)

for chunk in agent.execute(task, stream=True):
    if isinstance(chunk, AgentResult):
        print("\nFINAL ANSWER:\n\n")
        print(chunk.content)
    else:
        print(chunk)
    print("\n---------------------------------------------\n")