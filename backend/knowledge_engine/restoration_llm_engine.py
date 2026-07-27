import json
import re

from openai import OpenAI


class RestorationLLMEngine:

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    def generate(
        self,
        diagnosis,
        recommendations,
        species,
        restoration_brief,
    ):
        prompt = f"""
You are FYNOS AI's ecological restoration advisor.

The ecological analysis has ALREADY been completed by the FYNOS Restoration Intelligence Engine.

Your task is NOT to analyze the landscape again.

Using ONLY the structured Restoration Brief below, generate clear, concise and professional summaries.

Return ONLY valid JSON in the following format:

{{
    "impact_summary": "",
    "executive_summary": "",
    "farmer_guidance": ""
}}

Restoration Brief:

{json.dumps(restoration_brief, indent=2, ensure_ascii=False)}
"""

        DEBUG = False

        if DEBUG:
            print("\n" + "=" * 80)
            print("FYNOS LLM DEBUG")
            print("=" * 80)

            print("Diagnosis size:", len(json.dumps(diagnosis, default=str)))
            print("Recommendations size:", len(json.dumps(recommendations, default=str)))
            print("Species size:", len(json.dumps(species, default=str)))
            print("Restoration brief size:", len(json.dumps(restoration_brief, default=str)))

            prompt_size = len(prompt)

            print("Prompt characters:", prompt_size)
            print("Approx prompt tokens:", prompt_size // 4)

            print("=" * 80 + "\n")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )

        raw_text = response.output_text

        try:
            return json.loads(raw_text)

        except json.JSONDecodeError:

            match = re.search(r"\{.*\}", raw_text, re.DOTALL)

            if match:
                return json.loads(match.group())

            raise ValueError("No valid JSON returned by RestorationLLMEngine.")