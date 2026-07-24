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

        Using the information below, generate ONLY valid JSON.

        Context:

        Landscape Diagnosis:
        {diagnosis}

        Recommendations:
        {recommendations}

        Selected Species:
        {species}

        Restoration Brief:
        {restoration_brief}

        Return exactly:

        {{
            "impact_summary": "",
            "executive_summary": "",
            "farmer_guidance": ""
        }}
        """

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )

        import json
        import re

        raw_text = response.output_text

        try:
            return json.loads(raw_text)

        except json.JSONDecodeError:

            match = re.search(r"\{.*\}", raw_text, re.DOTALL)

            if match:
                return json.loads(match.group())

            raise ValueError("No valid JSON returned by RestorationLLMEngine.")