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
                "Do not include any explanations, comments, or code block markers. "
                "Only return the corrected YAML content. "
            )

            user_prompt = (
                f"The previous map.yaml caused an error during simulation:\n\n{error_message}\n\n"
                "Please regenerate a corrected map.yaml that avoids this issue while still following the hardware constraints defined in arch.yaml and problem.yaml.\n"
                "If the error is a constraint error, modify the factors by decreasing/increasing certain numbers so"
                "the constraint is no longer violated while making sure other constraints are still being followed.\n"
                f"\nArchitecture specification:\n{arch_yaml}"
                f"\nProblem description:\n{problem_yaml}"
                f"\nPrevious (invalid) map.yaml:\n{baseline_map}"
                "\nReturn only the YAML content without any additional text."
                "\nYou must follow these constraints:"
                "\n- For each level, tile size must be ≤ depth * block-size from arch.yaml."
                    "To calculate tile size, let *_eff be the cumulative product of ALL factors from that level (both spatial and temporal) and levels above it:"
                "\n       - Input tile size = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * ((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff"
                "\n       - Weight tile size = K_eff * R_eff * S_eff * C_eff * H_eff"
                "\n       - Output tile size = P_eff * Q_eff * K_eff * N_eff * H_eff"
                "\n       - for registers, use only weight tile size (weight tile size <= depth * block-size)"
                "\n       - for accumulation buffer, use only output tile size"
                "\n       - for weight buffer, use only weight tile size"
                "\n       - for input buffer, use only input tile size"
                "\n       - for global buffer, use the input tile size plus the output tile size"
                "\n       - example with baseline map: Mapping:"
                                "\n- Level 1 (Registers): Q=14 (temporal), N=8 (temporal), others=1"
                                "\n- Level 2 (AccumulationBuffer): P=7 (spatial), P=2 (temporal), C=2 (spatial), others=1"
                                "\n- Level 3 (WeightBuffer): K=16 (temporal), others=1"
                                "\nEffective dimensions at each level:"
                                "\n- At Level 1: R=1, S=1, P=1, Q=14, C=1, K=1, H=1, N=8"
                                "\n- At Level 2: R=1, S=1, P=14, Q=14, C=2, K=1, H=1, N=8"
                                "\n- At Level 3: R=1, S=1, P=14, Q=14, C=2, K=16, H=1, N=8"
                                "\n- Tile size for accumulation buffer is 14 * 14 * 1 * 8 * 1 = 1568 while buffer capacity is 768 * 4 = 3072"
                                
                "\n- Product of mapping factors per dimension (e.g., multiplying together each value of C at every level) must equal the instance size in problem.yaml"
                "\n- all types (spatial, temporal, datatype) must be included as they are in the baseline example map"
                "\n YOU MUST VERIFY ALL THESE CONSTRAINTS MATHEMATICALLY BEFORE RETURNING YOUR FINAL ANSWER"
            )
        else:
            system_prompt = (
                "You are an expert in creating tensor programs. "
                "Do not include any explanations, comments, or code block markers. "
                "Only return the corrected YAML content. "
            )

            user_prompt = (
                f"Hardware specification:\n{arch_yaml}"
                f"\nProblem description:\n{problem_yaml}"
                f"\nBaseline example map:\n{baseline_map}"
                "\nPlease suggest an optimized version of the example map (new map.yaml) "
                "that improves data locality and parallelization while also following the hardware constraints defined in arch.yaml and problem.yaml. "
                "Return only the YAML content without any additional text. "
                "You must follow these constraints:"
                "\n- For each level, tile size must be <= depth * block-size from arch.yaml."
                "To calculate tile size, let *_eff be the cumulative product of ALL factors from that level (both spatial and temporal) and levels above it:"
                "\n       - Input tile size = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * ((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff"
                "\n       - Weight tile size = K_eff * R_eff * S_eff * C_eff * H_eff"
                "\n       - Output tile size = P_eff * Q_eff * K_eff * N_eff * H_eff"
                "\n       - for registers, use only weight tile size (weight tile size <= depth * block-size)"
                "\n       - for accumulation buffer, use only output tile size"
                "\n       - for weight buffer, use only weight tile size"
                "\n       - for input buffer, use only input tile size"
                "\n       - for global buffer, use the input tile size plus the output tile size"
                "\n       - example with baseline map: Mapping:"
                                "\n- Level 1 (Registers): Q=14 (temporal), N=8 (temporal), others=1"
                                "\n- Level 2 (AccumulationBuffer): P=7 (spatial), P=2 (temporal), C=2 (spatial), others=1"
                                "\n- Level 3 (WeightBuffer): K=16 (temporal), others=1"
                                "\nEffective dimensions at each level:"
                                "\n- At Level 1: R=1, S=1, P=1, Q=14, C=1, K=1, H=1, N=8"
                                "\n- At Level 2: R=1, S=1, P=14, Q=14, C=2, K=1, H=1, N=8"
                                "\n- At Level 3: R=1, S=1, P=14, Q=14, C=2, K=16, H=1, N=8"
                                "\n- Tile size for accumulation buffer is 14 * 14 * 1 * 8 * 1 = 1568 while buffer capacity is 768 * 4 = 3072"
                                
                "\n- Product of mapping factors per dimension (e.g., multiplying together each value of C at every level) must equal the instance size in problem.yaml"
                "\n- all types (spatial, temporal, datatype) must be included as they are in the baseline example map"
                "\n YOU MUST VERIFY ALL THESE CONSTRAINTS MATHEMATICALLY BEFORE RETURNING YOUR FINAL ANSWER"
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
