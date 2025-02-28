import os
from dotenv import load_dotenv
import openai
# Load environment variables from .env file
from autogen import  AssistantAgent
from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent

from dotenv import load_dotenv

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities import teachability
from autogen import Cache
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
from autogen.agentchat.contrib.capabilities import transform_messages
import autogen
from autogen import ConversableAgent, UserProxyAgent
from typing import Literal, Annotated
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")



config_list = [
    {
        "model":"gpt-4",
        "temperature":0.9,
        "api_key":openai_api_key
    }
]

llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.ConversableAgent(
    name="Human",
    llm_config=False,
    human_input_mode='ALWAYS',
    is_termination_msg=lambda msg: "good bye" in msg['content'] or None
)


CurrencySymbol = Literal["USD", "EUR"]


def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")


@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"


print("Expected Function:", currency_calculator)
print("Actual Function:", user_proxy.function_map["currency_calculator"]._origin)
print("Are they the same object?", currency_calculator is user_proxy.function_map["currency_calculator"]._origin)

