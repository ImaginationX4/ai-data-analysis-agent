from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
import os
from typing import List, Tuple


class DataAnalysisAgent:
    """
    A conversational AI agent for data analysis using natural language.
    
    This agent can execute Python code to analyze CSV data, generate visualizations,
    and provide insights based on user queries in natural language.
    """
    
    def __init__(self, llm, csv_file_path: str):
        """
        Initialize the Data Analysis Agent.
        
        Args:
            llm: Language model instance (e.g., ChatOpenAI)
            csv_file_path (str): Path to the CSV file to analyze
        """
        self.llm = llm
        self.csv_file_path = csv_file_path
        self.chat_history: List[Tuple[str, str]] = []

        # Initialize Python execution tool
        self.python_tool = PythonREPLTool()
        self.tools = [self.python_tool]

        # Build prompt template for ReAct pattern
        self.prompt_template = PromptTemplate(
            input_variables=["tools", "chat_history", "tool_names", "query", "agent_scratchpad"],
            template='''Here is a CSV file located at {csv_file_path}. Please answer the following questions as best you can. You have access to the following tools:

            {tools}

            Chat history from previous conversations:
            {chat_history}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can be repeated zero or 3 times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {query}
            {agent_scratchpad}'''
        )

        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def _get_tool_descriptions(self) -> str:
        """
        Get descriptions of available tools.
        
        Returns:
            str: Tool descriptions for the agent
        """
        return "Python_REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."

    def chat(self, query: str) -> str:
        """
        Process user query and return analysis results.
        
        Args:
            query (str): User's natural language question about the data
            
        Returns:
            str: Agent's response with analysis results
        """
        print(f"\n🤖 User Question: {query}")

        try:
            # Prepare input parameters
            inputs = {
                "csv_file_path": self.csv_file_path,
                "tools": "Python_REPL",
                "chat_history": self.chat_history,
                "tool_names": self._get_tool_descriptions(),
                "query": query,
                "agent_scratchpad": "",
            }

            # Execute query
            response = self.agent_executor.invoke(inputs)

            # Save to chat history
            self.chat_history.append((query, response['output']))

            return response['output']

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            print(f"\n❌ Error: {error_msg}")

            # Add error to history for future reference
            self.chat_history.append((query, error_msg))
            return error_msg

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        print("✅ Chat history cleared.")

    def get_history(self) -> List[Tuple[str, str]]:
        """
        Get current chat history.
        
        Returns:
            List[Tuple[str, str]]: List of (question, answer) pairs
        """
        return self.chat_history