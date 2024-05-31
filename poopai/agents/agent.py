from typing import Generator, List, Optional, Union
from pydantic import BaseModel, Field, PrivateAttr, validator
import textwrap
import json

from poopai.drivers.base_driver import BaseDriver
from poopai.drivers.openai_driver import OpenAIDriver
from poopai.tools import BaseTool
from poopai.tools.default_agent_tools import _AgentFinishTool


class AgentResult(BaseModel):
    content: str = Field("Final result of agent execution")


class Agent(BaseModel):
    driver: BaseDriver
    description: str = Field("Description of AI agent role")
    tools: Optional[List[BaseTool]] = Field(default=None)
    max_iterations: int = Field(default=10, ge=0)
    _scratchpad: List[str] = PrivateAttr(default=[])


    @validator('tools', pre=True)
    def add_default_tools(cls, value):
        """Add default tools to tool list"""
        if not value:
            value = [_AgentFinishTool()]
        else:
            value.append(_AgentFinishTool())
        return value
    

    def directed_edge(self, agent: 'Agent'):
        # Check if arg is instance of Agent class
        if not isinstance(agent, Agent):
            raise ValueError("Argument must be an instance of Agent.")
        raise NotImplementedError("Method not implemented.")
    

    def execute(self, task: str, input: Optional[str] = None, stream: bool = False) -> Union[Generator, AgentResult]:
        # Add initial data to conversation history and agent scratchpad
        self._scratchpad.append(f"task: {task}")
        if input:
            self._scratchpad.append(input)

        # Loop through iterations
        for _ in range(self.max_iterations):
            for chunk in self._iterate():
                if stream:
                    yield chunk
                if isinstance(chunk, AgentResult):
                    return chunk

        # At this point, maximum number of iterations reached
        raise RuntimeError("Maximum number of iterations reached.")
    
    
    def refresh_memory(self):
        memory = f"""
            YOU ARE AN AI AGENT GIVEN THE FOLLOWING ROLE:
            {self.description}

            YOU ARE TO COMPLETE THE TASK BASED ON THE FOLLOWING PROGRESS:
            {self._scratchpad}
        """
        return textwrap.dedent(memory)
    

    def _iterate(self) -> Generator:
        memory = self.refresh_memory()
        plan = self.driver._plan(
            system_prompt=memory,
            tools=self.tools,
        )
        self._scratchpad.append(f"plan: {plan.content}")
        yield plan

        memory = self.refresh_memory()
        action = self.driver._action(
            system_prompt=memory,
            tools=self.tools,
        )
        tool_calls = action.tool_calls
        json_tool_calls = []
        for tool_call in tool_calls:
            json_tool_calls.append(tool_call.json())
            if tool_call.name == _AgentFinishTool().name:
                result = json.loads(tool_call.args)["result"]
                yield AgentResult(content=result)
                return
        self._scratchpad.append(f"action(s): Calling the following tools - {json_tool_calls}")
        yield action
        
        memory = self.refresh_memory()
        observation = self.driver._observe(
            system_prompt=memory,
            tools=self.tools,
            action=action,
        )
        self._scratchpad.append(f"observation(s): {observation.content}")
        yield observation
            

            # def _plan(
    #     self, 
    #     system_prompt: str, 
    #     tools: List[BaseTool],
    # ) -> DriverPlan:
    #     tools_json_desc = [tool.get_schema().json() for tool in tools]
    #     system_prompt = f"""
    #         {system_prompt}

    #         YOU CAN USE THE FOLLOWING TOOLS:
    #         {tools_json_desc}

    #         WRITE A BRIEF PLAN FOR WHAT YOU SHOULD DO AT THIS POINT IN TIME:
    #     """
    #     system_prompt = textwrap.dedent(system_prompt)
    #     messages = [{"role": "system", "content": system_prompt}]
    #     response = self._client.chat.completions.create(
    #         model=self.model,
    #         messages=messages,
    #     )
    #     content = response.choices[0].message.content
    #     usage = self._openai_usage_to_driver_usage(response.usage)
    #     return DriverPlan(content=content, usage=usage)
    
    
    # def _action(
    #     self, 
    #     system_prompt: str, 
    #     tools: List[BaseTool],
    # ) -> DriverAction:
    #     system_prompt = f"""
    #         {system_prompt}

    #         USE THE GIVEN TOOLS TO EXECUTE YOUR PLAN.
    #     """
    #     system_prompt = textwrap.dedent(system_prompt)
    #     messages = [{"role": "system", "content": system_prompt}]
    #     openai_functions = [self._basetool_to_openai_fc_schema(tool) for tool in tools]
    #     response = self._client.chat.completions.create(
    #         model=self.model,
    #         messages=messages,
    #         tools=openai_functions,
    #         tool_choice="required",
    #     )
    #     tool_calls = response.choices[0].message.tool_calls 
    #     driver_tool_calls = []
    #     for tool_call in tool_calls:
    #         driver_tool_call = DriverToolCall(id=tool_call.id, name=tool_call.function.name, args=tool_call.function.arguments)
    #         driver_tool_calls.append(driver_tool_call)
    #     usage = self._openai_usage_to_driver_usage(response.usage)
    #     return DriverAction(usage=usage, tool_calls=driver_tool_calls)


    # def _observe(
    #     self, 
    #     system_prompt: str,
    #     tools: List[BaseTool],
    #     action: DriverAction
    # ) -> DriverObservation:
    #     data = []
    #     for tool_call in action.tool_calls:
    #         # Get function call info
    #         function_name = tool_call.name
    #         function_args = json.loads(tool_call.args)

    #         # Iterate through provided tools to check if driver_response function call matches one
    #         no_match_flag = True
    #         for tool in tools:
    #             # If match, run tool function on arguments for result, and append to memory
    #             if tool.get_schema().name == function_name:
    #                 no_match_flag = False
    #                 function_result = str(tool.func(**function_args))
    #                 data.append(function_result)
            
    #         # If driver_response function call matches none of the given tools
    #         if no_match_flag:
    #             raise Exception("Driver called function, function call does not match any of the provided tools.")
            
    #     # Once data has been obtained from the results of function calls, filter for useful insights as observations
    #     system_prompt = f"""
    #         {system_prompt}

    #         HERE ARE IS WHAT YOU OBSERVED FROM YOUR PREVIOUS ACTION(S):
    #         {data}

    #         EXTRACT THE MOST RELEVANT DATA FROM YOUR OBSERVATIONS:
    #     """
    #     system_prompt = textwrap.dedent(system_prompt)
    #     messages = [{"role": "system", "content": system_prompt}]
    #     response = self._client.chat.completions.create(
    #         model=self.model,
    #         messages=messages,
    #     )
    #     content = response.choices[0].message.content
    #     usage = self._openai_usage_to_driver_usage(response.usage)
    #     return DriverObservation(content=content, usage=usage)