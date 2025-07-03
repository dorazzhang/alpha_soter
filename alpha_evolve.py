import anthropic
import subprocess
import os
import yaml
import re

class LLMOptimizer:
    def __init__(self, api_key, model="claude-sonnet-4-20250514", max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("Anthropic API key must be provided.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def optimize(self, arch_yaml, problem_yaml, baseline_map, regenerate = False, error_message = None):

        if regenerate:
            system_prompt = (
                "You are an expert in creating tensor programs. "
                "Regenerate a valid YAML map configuration (map.yaml) that fixes the issues in the previous attempt. "
                "Do not include any explanations, comments, or code block markers. "
                "Only return the corrected YAML content. "
                "Follow hardware constraints defined in arch.yaml and problem.yaml."
            )

            user_prompt = (
                f"The previous map.yaml caused an error during simulation:\n\n{error_message}\n\n"
                "Please regenerate a corrected map.yaml that avoids this issue.\n"
                f"\nArchitecture specification:\n{arch_yaml}"
                f"\nProblem description:\n{problem_yaml}"
                f"\nPrevious (invalid) map.yaml:\n{baseline_map}"
                "Remember, return only the YAML content without any additional text."
            )
        else:
            system_prompt = (
                "You are an expert in creating tensor programs. "
                "You must respond with ONLY a valid YAML map configuration. "
                "Do not include any explanations, comments, or code block markers. "
                "Start your response immediately with the YAML content. "
                "Ensure proper YAML formatting with correct indentation and syntax. "
                "Follow constraints outlined in the arch.yaml and problem.yaml."
            )

            user_prompt = (
                f"Hardware specification:\n{arch_yaml}"
                f"\nProblem description:\n{problem_yaml}"
                f"\nBaseline example map:\n{baseline_map}"
                "\nPlease suggest an optimized version of the example map (new map.yaml) "
                "that improves data locality and parallelization. "
                "Return only the YAML content without any additional text. "
                "Follow these constraints:"
                "\n- For each level, tile size must be ≤ depth × block-size from arch.yaml"
                "\n- Product of mapping factors per dimension (e.g., C) must equal the instance size in problem.yaml"
            )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
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
