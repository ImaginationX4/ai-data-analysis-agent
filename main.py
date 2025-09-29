#!/usr/bin/env python3
"""
Main entry point for the Data Analysis Agent.

This script demonstrates how to use the DataAnalysisAgent to interact with CSV data
using natural language queries.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent import DataAnalysisAgent


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    
    return api_key


def initialize_llm(api_key: str, model: str = "gpt-4o", temperature: float = 0):
    """
    Initialize the language model.
    
    Args:
        api_key (str): OpenAI API key
        model (str): Model name to use
        temperature (float): Model temperature setting
        
    Returns:
        ChatOpenAI: Initialized language model
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def print_welcome_message():
    """Print welcome message and usage examples."""
    print("🚢 Data Analysis Agent Started!")
    print("=" * 50)
    print("You can ask questions in natural language, such as:")
    print("- 'Show me basic statistics of the data'")
    print("- 'Create a survival rate distribution chart'")
    print("- 'Predict survival rate using age and tell me the accuracy'")
    print("- 'What are the correlations between different features?'")
    print("- 'Show me the distribution of passenger classes'")
    print("=" * 50)


def main():
    """Main function to run the interactive data analysis agent."""
    try:
        # Load environment variables
        api_key = load_environment()
        
        # Initialize language model
        llm = initialize_llm(api_key)
        
        # CSV file path (modify this according to your data file)
        csv_file_path = "data.csv"  # Change this to your CSV file path
        
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            print(f"❌ Error: CSV file '{csv_file_path}' not found.")
            print("Please ensure your data file exists and update the path in main.py")
            return
        
        # Create data analysis agent
        agent = DataAnalysisAgent(
            llm=llm,
            csv_file_path=csv_file_path
        )
        
        # Print welcome message
        print_welcome_message()
        
        # Start interactive loop
        while True:
            try:
                user_input = input("\n💬 Enter your question (type 'quit' to exit): ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['clear', 'clear history']:
                    agent.clear_history()
                    continue
                
                if user_input.lower() in ['history', 'show history']:
                    history = agent.get_history()
                    if history:
                        print("\n📝 Chat History:")
                        for i, (q, a) in enumerate(history, 1):
                            print(f"{i}. Q: {q}")
                            print(f"   A: {a[:100]}..." if len(a) > 100 else f"   A: {a}")
                    else:
                        print("No chat history yet.")
                    continue
                
                if not user_input.strip():
                    print("Please enter a valid question.")
                    continue
                
                # Process the query
                response = agent.chat(user_input)
                print(f"\n📊 Agent Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("Please try again or restart the application.")

    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()