import os
import json
from inputimeout import inputimeout, TimeoutOccurred

from core.llm import GroqLLM
from config.config import Config
from core.logger import log_interaction, get_session_history

def main():
    os.makedirs(Config.DATA_FOLDER, exist_ok=True)
    os.makedirs(Config.LOG_FOLDER, exist_ok=True)

    llm = GroqLLM()
    messages = [
        {"role": "system", "content": Config.SYSTEM_PROMPT}
    ]
    print(
        "Welcome! Example: 'show my cart', 'add <product_id> to cart', "
        "'show electronics under 40000'.\n"
        "Type 'exit' to quit. Type 'show history' to view all session history."
    )
    while True:
        try:
            user_input = inputimeout(prompt="You: ", timeout=Config.TIMEOUT_SECONDS)
        except TimeoutOccurred:
            print("\nSession timed out. Exiting.")
            break

        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        if user_input.strip().lower() == "show history":
            history = get_session_history()
            print(json.dumps(history, indent=2))
            continue

        messages.append({"role": "user", "content": user_input})
        llm_response = llm.generate(messages)
        messages.append({"role": "assistant", "content": json.dumps(llm_response)})
        print(f"LLM (structured): {json.dumps(llm_response, indent=2)}")

        # Log each turn (no user ID required)
        log_interaction({
            "user_input": user_input,
            "llm_response": llm_response
        })

if __name__ == "__main__":
    main()
