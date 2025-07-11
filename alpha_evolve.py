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
                "Only return the corrected YAML file starting with \'mapping:\'. "
            )

            user_prompt = (
                f"The previous map.yaml caused an error during simulation:\n\n{error_message}\n\n"
                "Task:\n\n"
                "Please regenerate a corrected map.yaml that avoids this issue while still following the hardware constraints defined in arch.yaml and problem.yaml.\n"
                "If the error is a constraint error, modify the factors by decreasing/increasing certain numbers so"
                "the constraint is no longer violated while making sure other constraints are still being followed.\n"

                "Here's what you should do to correct the following errors:\n"
                "1. Mapping Format: add the missing section (e.g., the spatial type for Accumulation Buffer) to the mapping and update factors accordingly\n"
                "2. Mapped tile size exceeds buffer capacity. Decrease factors at or before the level that contribute to the tile size. (e.g., If the Accumlation Buffer's tile size is the problem, try decreasing dimension P at that level.)\n"
                "3. product of all factors of a dimension is not equal to the dimension size of the workload. Increase or decrease factors of that dimension at different levels until they all multiply to the instance size\n\n"
                "YAML files:\n\n"
                f"Hardware specification:\n{arch_yaml}\n\n"
                f"Problem description:\n{problem_yaml}\n\n"
                f"Baseline example map:\n{baseline_map}\n\n"

                "Previous Task:\n"
                "Generate an optimized map.yaml that improves data locality and parallelism "
                "while adhering to hardware constraints defined in arch.yaml and problem.yaml.\n\n"

                "Previous Instructions:\n"
                "- Return only the YAML content of the optimized map.yaml (no prose or explanations).\n"
                "- Your mapping must strictly satisfy all hardware and problem constraints, verified using the following rules:\n\n"

                "Verification checklist (with calculation examples using the baseline map attached):\n\n"

                "1. Mapping Format:\n"
                "- Includes all required mapping levels and types (temporal, spatial, datatype), as shown in the baseline map.\n\n"

                "2. Buffer Tile Size Constraints (based on arch.yaml):\n"
                "- Let *_eff denote the product of all mapping factors for a dimension up to a given level.\n\n"

                "  Registers (weight tile):\n"
                "    Tile size = K_eff * R_eff * S_eff * C_eff * H_eff (e.g., 1*1*1*1*1 = 1)\n\n"

                "  Accumulation Buffer (output tile size):\n"
                "    Tile size = P_eff * Q_eff * K_eff * N_eff * H_eff "
                "(e.g., 14*14*1*8*1 = 1568)\n\n"

                "  Weight Buffer (weight tile size):\n"
                "    Tile size = K_eff * R_eff * S_eff * C_eff * H_eff "
                "(e.g., 16*1*1*2*1 = 32)\n\n"

                "  Input Buffer (input tile size):\n"
                "    Tile size = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * "
                "((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff "
                "(e.g., 16 * 13 * 13 * 2 * 1 = 6272)\n\n"

                "  Global Buffer (input + output tile sizes):\n"
                "    Input tile = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * "
                "((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff (e.g., 16 * ((14 - 1) * 1 + 1 + 1 * (1 - 1)) * ((14 - 1) * 1 + 1 + 1 * (1 - 1)) * 4 * 1 = 12544) \n"
                "    Output tile = P_eff * Q_eff * K_eff * N_eff * H_eff (e.g., 14 * 14 * 16 * 1 = 3584)\n"
                "    Total = input + output = 12544 + 3584 = 9856\n\n"

                "- Ensure all tile sizes are <= depth * block-size from arch.yaml at each level.\n\n"

                "3. Problem Coverage (based on problem.yaml):\n"
                "- The product of mapping factors per dimension (across all levels) must match the instance size. For example:\n"
                "    R_eff = 1*1*1*1*1*1*1*1 = 1 == Instance size = 1\n"
                "    P_eff = 1*7*2*1*1*1*1*1 = 14 == Instance size = 14\n"
                "    C_eff = 1*1*2*1*1*1*2*256 = 1024 == Instance size = 1024\n"
                "(Continue verifying: S, Q, K, H, N...)\n\n"

                "Important:\n"
                "- Do not return until all constraints are mathematically verified in your solution.\n"
                "- Return only the final optimized map.yaml content with no other text, starting with \'mapping:\'.\n"
            )

        else:
            system_prompt = (
                "You are an expert in creating tensor programs. "
                "Do not include any explanations, comments, or code block markers. "
                "Only return the corrected YAML content starting with \'mapping:\'. "
            )

            user_prompt = (
                "YAML files:\n\n"
                f"Hardware specification:\n{arch_yaml}\n\n"
                f"Problem description:\n{problem_yaml}\n\n"
                f"Baseline example map:\n{baseline_map}\n\n"

                "Task:\n"
                "Generate an optimized map.yaml that improves data locality and parallelism "
                "while adhering to hardware constraints defined in arch.yaml and problem.yaml.\n\n"

                "Instructions:\n"
                "- Return only the YAML content of the optimized map.yaml (no prose or explanations).\n"
                "- Your mapping must strictly satisfy all hardware and problem constraints, verified using the following rules:\n\n"

                "Verification checklist (with calculation examples using the baseline map attached):\n\n"

                "1. Mapping Format:\n"
                "- Includes all required mapping levels and types (temporal, spatial, datatype), as shown in the baseline map.\n\n"

                "2. Buffer Tile Size Constraints (based on arch.yaml):\n"
                "- Let *_eff denote the product of all mapping factors for a dimension up to a given level.\n\n"

                "  Registers (weight tile):\n"
                "    Tile size = K_eff * R_eff * S_eff * C_eff * H_eff (e.g., 1*1*1*1*1 = 1)\n\n"

                "  Accumulation Buffer (output tile size):\n"
                "    Tile size = P_eff * Q_eff * K_eff * N_eff * H_eff "
                "(e.g., 14*14*1*8*1 = 1568)\n\n"

                "  Weight Buffer (weight tile size):\n"
                "    Tile size = K_eff * R_eff * S_eff * C_eff * H_eff "
                "(e.g., 16*1*1*2*1 = 32)\n\n"

                "  Input Buffer (input tile size):\n"
                "    Tile size = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * "
                "((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff "
                "(e.g., 16 * 13 * 13 * 2 * 1 = 6272)\n\n"

                "  Global Buffer (input + output tile sizes):\n"
                "    Input tile = N_eff * ((P_eff-1)*wstride + 1 + wdilation*(R_eff-1)) * "
                "((Q_eff-1)*hstride + 1 + hdilation*(S_eff-1)) * C_eff * H_eff (e.g., 16 * ((14 - 1) * 1 + 1 + 1 * (1 - 1)) * ((14 - 1) * 1 + 1 + 1 * (1 - 1)) * 4 * 1 = 12544) \n"
                "    Output tile = P_eff * Q_eff * K_eff * N_eff * H_eff (e.g., 14 * 14 * 16 * 1 = 3584)\n"
                "    Total = input + output = 12544 + 3584 = 9856\n\n"

                "- Ensure all tile sizes are <= depth * block-size from arch.yaml at each level.\n\n"

                "3. Problem Coverage (based on problem.yaml):\n"
                "- The product of mapping factors per dimension (across all levels) must match the instance size. For example:\n"
                "    R_eff = 1*1*1*1*1*1*1*1 = 1 == Instance size = 1\n"
                "    P_eff = 1*7*2*1*1*1*1*1 = 14 == Instance size = 14\n"
                "    C_eff = 1*1*2*1*1*1*2*256 = 1024 == Instance size = 1024\n"
                "(Continue verifying: S, Q, K, H, N...)\n\n"

                "Important:\n"
                "- Do not return until all constraints are mathematically verified in your solution.\n"
                "- Return only the final optimized map.yaml content with no other text, starting with \'mapping:\'.\n"
                
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
