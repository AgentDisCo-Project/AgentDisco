"""
This module contains the generator components. Generators are commonly used in LLM Applications to generate responses
based on a given input via a model service. The type of input and output can vary depending on the generator. For
example, a chat generator takes a chat prompt (a list of ChatMessages) as input and generate a list of response
candidates as output.
"""

from allin_llmflow.components.generators.chat_generator import ChatGenerator

__all__ = [
    "ChatGenerator",
]
