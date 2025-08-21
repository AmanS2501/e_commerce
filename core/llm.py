import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from config.config import Config

load_dotenv()

RESPONSE_SCHEMAS = [
    ResponseSchema(
        name="action",
        description="The requested action. Required. One of: show_cart, add_to_cart, remove_from_cart, search_products, get_order_status, reply"
    ),
    ResponseSchema(
        name="id",
        description="Product ID for add_to_cart/remove_from_cart actions. Optional, null if not used."
    ),
    ResponseSchema(
        name="filters",
        description="Filters for product search. Optional, null if not used."
    ),
    ResponseSchema(
        name="order_id",
        description="Order ID for get_order_status. Optional, null if not used."
    ),
    ResponseSchema(
        name="message",
        description="The reply message for reply actions. Optional, null if not used."
    )
]
output_parser = StructuredOutputParser.from_response_schemas(RESPONSE_SCHEMAS)

class GroqLLM:
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            max_retries=2
        )

    def generate(self, messages):
        raw_response = self.llm.invoke(messages).content
        try:
            structured = output_parser.parse(raw_response)
            return structured
        except Exception as e:
            return {"action": "reply", "message": f"Unable to parse response: {raw_response} (Error: {e})"}
