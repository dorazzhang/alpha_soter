import anthropic
import os
import yaml
import re

class LLMOptimizer:
    def __init__(self, api_key, model="claude-opus-4-20250514", max_tokens=1000, temperature=0.7):
        """
        :param api_key: Your Anthropics API key (string).
        :param model: Model name, e.g., "claude-opus-4-20250514".
        :param max_tokens: Max tokens for the response.
        :param temperature: Model temperature (0–2).
        """
        if not api_key:
            raise ValueError("Anthropic API key must be provided.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def optimize(self, arch_yaml, problem_yaml, baseline_map):
        """
        Given a baseline tensor program and optimization instructions,
        returns Claude's proposed optimized version.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system= "You are an expert in creating tensor programs. "
            "You must respond with ONLY a valid YAML map configuration. "
            "Do not include any explanations, comments, or code block markers. "
            "Start your response immediately with the YAML content. "
            "Ensure proper YAML formatting with correct indentation and syntax.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (

                                f"Here is the hardware specification:\n{arch_yaml}"
                                f"\nHere is the problem description:\n{problem_yaml}"
                                f"\nHere is a baseline example map:\n{baseline_map}"
                                "\nPlease suggest an optimized version of the example map (new map.yaml) "
                                "that improves data locality and parallelization. "
                                "Return only the YAML content without any additional text."
                            )
                        }
                    ]
                }
            ]
        )

        try:
            yaml.safe_load(message.content[0].text)
            return message.content[0].text
        except yaml.YAMLError:
            print(f"Invalid YAML generated. Error: \n{message.content[0].text}")
            return None
