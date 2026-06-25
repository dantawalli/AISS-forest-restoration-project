import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for Structured Output ---

class RecommendationItem(BaseModel):
    recommendation_number: int
    title: str
    text: str = Field(description="The full detailed recommendation content including Objective, Actions, Timeframe, and Evidence.")

class ForestAnalysisResponse(BaseModel):
    summary: str = Field(description="A high-level executive summary of the deforestation situation and data trends.")
    recommendations: List[RecommendationItem]

# --- Main Class ---

class ForestRecommendationEngine:
    def __init__(self, api_key: str, df: pd.DataFrame):
        """Initialize recommendation engine with OpenAI API and data"""
        self.client = OpenAI(api_key=api_key)
        # Using GPT-4 for structured outputs and reliability
        self.model = 'gpt-4o'
        self.df = df
        self.cache = {}
    
    def _json_safe(self, obj):
        """Recursively convert NumPy/Pandas types to JSON-safe Python types"""
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        else:
            return obj
        
    def get_forest_data(self, country: str, start_year: int, end_year: int) -> Dict[str, Any]:
        """Fetch historical forest data for a country"""
        country_data = self.df[
            (self.df['country'] == country) & 
            (self.df['year'] >= start_year) & 
            (self.df['year'] <= end_year)
        ].copy()
        
        if country_data.empty:
            return {}
        
        numeric_cols = ['tree_cover_loss_ha', 'primary_forest_loss_ha', 'carbon_gross_emissions_MgCO2e']
        for col in numeric_cols:
            if col in country_data.columns:
                country_data[col] = pd.to_numeric(country_data[col], errors='coerce').fillna(0)
        
        yearly_loss = country_data.groupby('year')['tree_cover_loss_ha'].sum().reset_index()
        
        if len(yearly_loss) >= 2:
            years = yearly_loss['year'].values.reshape(-1, 1)
            losses = yearly_loss['tree_cover_loss_ha'].values
            lr = LinearRegression().fit(years, losses)
            trend_rate = lr.coef_[0] / losses.mean() * 100 if losses.mean() > 0 else 0
            trend_direction = "increasing" if trend_rate > 0 else "decreasing"
        else:
            trend_rate = 0
            trend_direction = "stable"
        
        return {
            'total_loss': country_data['tree_cover_loss_ha'].sum(),
            'yearly_data': yearly_loss.to_dict('records'),
            'trend_direction': trend_direction,
            'trend_rate': abs(trend_rate),
            'years_analyzed': len(yearly_loss),
            'primary_forest_loss': country_data['primary_forest_loss_ha'].sum(),
            'carbon_emissions': country_data['carbon_gross_emissions_MgCO2e'].sum()
        }
    
    def get_predictions(self, country: str, years_ahead: int = 5) -> Dict[str, Any]:
        """Get predictions from pre-calculated data"""
        try:
            # Note: Path adjustment might be needed based on your local env
            predictions_path = Path("data/predicted_tree_cover_loss_2025_2035.csv")
            
            if not predictions_path.exists():
                return {'projected_loss': 0, 'confidence': 0, 'risk_areas': [], 'yearly_predictions': []}
            
            pred_df = pd.read_csv(predictions_path)
            country_predictions = pred_df[pred_df.iloc[:, 0] == country]
            
            if country_predictions.empty:
                return {'projected_loss': 0, 'confidence': 0, 'risk_areas': [], 'yearly_predictions': []}
            
            years_to_use = min(years_ahead, len(country_predictions))
            yearly_predictions = []
            projected_loss = 0
            
            for i in range(years_to_use):
                row = country_predictions.iloc[i]
                loss = max(0, float(row.iloc[2]))
                yearly_predictions.append({'year': int(row.iloc[1]), 'predicted_loss': loss})
                projected_loss += loss
            
            return {
                'projected_loss': projected_loss,
                'confidence': 75.0,
                'risk_areas': ["continued_loss_expected"] if projected_loss > 0 else [],
                'yearly_predictions': yearly_predictions
            }
        except Exception as e:
            logger.error(f"Error reading predictions: {str(e)}")
            return {'projected_loss': 0, 'confidence': 0, 'risk_areas': [], 'yearly_predictions': []}

    def analyze_deforestation_drivers(self, country: str) -> Dict[str, Any]:
        """Analyze deforestation drivers for a country"""
        country_data = self.df[self.df['country'] == country].copy()
        if country_data.empty: return {}
        
        driver_mapping = {
            'hard_commodities': 'Agriculture', 'logging': 'Logging', 'wildfire': 'Wildfire',
            'permanent_agriculture': 'Agriculture', 'shifting_cultivation': 'Agriculture',
            'settlements_infrastructure': 'Infrastructure', 'other_natural_disturbances': 'Other'
        }
        
        driver_totals = {}
        for col, driver_name in driver_mapping.items():
            if col in country_data.columns:
                val = pd.to_numeric(country_data[col], errors='coerce').fillna(0).sum()
                driver_totals[driver_name] = driver_totals.get(driver_name, 0) + val
        
        sorted_drivers = sorted(driver_totals.items(), key=lambda x: x[1], reverse=True)
        return {'primary': [d for d, _ in sorted_drivers[:3]], 'all_drivers': dict(sorted_drivers)}

    def get_regional_benchmarks(self, country: str) -> Dict[str, Any]:
        """Get regional benchmarks for a country"""
        regional_groups = {
            'South America': ['Brazil', 'Argentina', 'Bolivia', 'Colombia', 'Peru', 'Venezuela', 'Ecuador', 'Guyana', 'Suriname', 'Paraguay', 'Chile'],
            'Southeast Asia': ['Indonesia', 'Malaysia', 'Thailand', 'Vietnam', 'Philippines', 'Cambodia', 'Lao People\'s Democratic Republic', 'Myanmar', 'Papua New Guinea']
        }
        
        region = next((r for r, cs in regional_groups.items() if country in cs), 'Other')
        region_countries = regional_groups.get(region, [country])
        region_data = self.df[self.df['country'].isin(region_countries)]
        
        country_losses = region_data.groupby('country')['tree_cover_loss_ha'].sum()
        country_loss = country_losses.get(country, 0)
        
        return {
            'regional_average': country_losses.mean(),
            'rank': (country_losses > country_loss).sum() + 1,
            'total_countries': len(country_losses),
            'region': region
        }

    def build_recommendation_context(self, country: str, stakeholder: str, data_range: Dict[str, int]) -> Dict[
        str, Any]:
        historical_data = self.get_forest_data(country, data_range['startYear'], data_range['endYear'])
        predictions = self.get_predictions(country, years_ahead=5)
        drivers = self.analyze_deforestation_drivers(country)
        benchmarks = self.get_regional_benchmarks(country)

        context = {
            'country': country,
            'stakeholder': stakeholder,
            'data_range': data_range,
            'historical_data': historical_data,
            'predictions': predictions,
            'drivers': drivers,
            'benchmarks': benchmarks,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

        return context

    def generate_stakeholder_prompt(self, context: Dict[str, Any]) -> str:
        """Generate stakeholder-specific prompt for LLM"""

        stakeholder_guidance = {
            "policy_governance":
                "Focus on regulation, public policy, enforcement, land-use planning, governance, government investment, and institutional capacity.",

            "academic_research":
                "Focus on scientific research priorities, monitoring frameworks, data collection, research partnerships, publication opportunities, knowledge gaps, and methodological improvements.",

            "environmental_ngo":
                "Focus on community engagement, conservation programs, indigenous partnerships, capacity building, biodiversity protection, grant-funded initiatives, and local participation.",

            "corporate_sustainability":
                "Focus on ESG strategies, carbon credits, supply-chain sustainability, corporate reporting, certification programs, net-zero targets, sustainable sourcing, and business partnerships."
        }

        base_prompt = f"""
    Generate EXACTLY 5 detailed, evidence-based recommendations for {context['stakeholder']} regarding {context['country']}'s forest loss.

    STAKEHOLDER CONTEXT:
    {stakeholder_guidance.get(context['stakeholder'], '')}

STAKEHOLDER-SPECIFIC DECISION AUTHORITY:

Policy & Governance:
- Focus on laws, regulations, enforcement, public investment, land-use planning, government programs, institutional reforms, and national strategies.
- Recommendations must involve actions that governments can directly implement.
- Do not recommend actions that depend primarily on corporations, NGOs, or academic institutions.

Academic & Research:
- Focus on scientific studies, monitoring methodologies, research networks, publications, field experiments, innovation, data systems, and knowledge generation.
- Recommendations must involve research, scientific collaboration, technology development, monitoring systems, or evidence generation.
- Avoid recommendations centered on policy enforcement, subsidies, or corporate procurement decisions.

Environmental NGO:
- Focus on community engagement, indigenous partnerships, awareness campaigns, training programs, conservation projects, biodiversity protection, and grant-funded initiatives.
- Recommendations must involve activities NGOs can directly coordinate or fund.
- Avoid recommendations centered on legislation, government enforcement, or corporate procurement requirements.

Corporate Sustainability:
- Focus on ESG programs, supply-chain management, certification schemes, carbon markets, supplier compliance, sustainable sourcing, reporting, and private-sector partnerships.
- Recommendations must involve actions corporations can directly implement.
- Prioritize:
  • supplier engagement,
  • procurement standards,
  • traceability systems,
  • certification,
  • ESG reporting,
  • carbon markets,
  • sustainable finance,
  • investment decisions,
  • supply-chain transformation.
- Do not recommend:
  • national legislation,
  • government subsidies,
  • public-sector enforcement,
  • nationwide agricultural programs,
  • public land-use planning,
  • government-led restoration initiatives.

IMPORTANT:
- Recommendations must reflect the actual decision-making authority of the selected stakeholder.
- Do not propose actions primarily controlled by another stakeholder group.
- Recommendations should sound substantially different across stakeholders.

STAKEHOLDER VALIDATION CHECK:

Before generating recommendations:

- Ask whether the stakeholder can directly implement the recommendation.
- If the stakeholder cannot directly control the action, revise it.
- Recommendations that require another stakeholder's authority are invalid.

CRITICAL STAKEHOLDER AUTHORITY TEST

Before finalizing each recommendation:

1. Ask: Can this stakeholder directly implement this action?

2. If the answer is NO, rewrite the recommendation.

Examples:

Corporate Sustainability:
- Can create supplier standards.
- Cannot pass laws.
- Cannot enforce national regulations.
- Cannot run government agricultural programs.

Policy & Governance:
- Can create laws and incentives.
- Can enforce regulations.
- Cannot directly manage corporate supply chains.
- Does not operate NGO conservation projects.

Environmental NGO:
- Can train communities.
- Can run conservation projects.
- Can advocate for policy.
- Cannot enact laws.

Academic & Research:
- Can conduct studies.
- Can develop methodologies.
- Can build monitoring systems.
- Cannot implement national policy.

    DATA CONTEXT:
- Total Historical Loss: {context['historical_data'].get('total_loss', 0):,} ha
- Annual Trend: {context['historical_data'].get('trend_rate', 0):.1f}% ({context['historical_data'].get('trend_direction')})
- Primary Forest Loss: {context['historical_data'].get('primary_forest_loss', 0):,} ha
- CO2 Emissions: {context['historical_data'].get('carbon_emissions', 0):,.0f} Mg
- Drivers: {', '.join(context['drivers'].get('primary', []))}
- 5-Year Projection: {
    context['predictions'].get('projected_loss')
    if context['predictions'].get('confidence', 0) > 0
    else 'Not Available'
}
- Prediction Confidence: {context['predictions'].get('confidence', 0)}
- Region: {context['benchmarks'].get('region', 'Unknown')}

DRIVER BREAKDOWN:
{json.dumps(context['drivers'].get('all_drivers', {}), indent=2)}

IMPORTANT DRIVER ANALYSIS:

- Identify the largest, second-largest, and third-largest drivers.
- Calculate relative differences between drivers whenever possible.
- Explain why the largest driver deserves priority.
- Use exact values from DRIVER BREAKDOWN.
- Do not use vague statements such as "Agriculture is important".
- Quantify driver dominance using the provided numbers.

Before generating recommendations:

1. Identify the largest driver.
2. Identify the second-largest driver.
3. Calculate the ratio between them.
4. Use these calculations when prioritizing recommendations.
5. Explain why the largest driver deserves intervention priority.

REGIONAL BENCHMARKS:
- Regional Rank: {context['benchmarks'].get('rank')}
- Regional Average Loss: {context['benchmarks'].get('regional_average'):,.0f} ha

CONSTRAINTS:
- No conversational filler or introductions.
- Every recommendation must be highly specific and evidence-based.
- Every recommendation must explicitly reference at least TWO numerical values from the provided data.
- Avoid generic sustainability advice.
- Explain WHY the recommendation is needed using the provided evidence.
- Recommendations should be practical and suitable for implementation by the selected stakeholder.
- The Objective section must contain 50-100 words.
- The Objective section MUST contain at least 50 words.
- Objectives shorter than 50 words are invalid.
- The Objective must explain:
  • why the recommendation is needed,
  • which forest-loss driver it addresses,
  • how the recommendation relates to the provided data,
  • what outcome is expected.
- Include at least 5 Specific Actions.
- Required Resources should be detailed and realistic.
- Expected Measurable Impact must include measurable outcomes and numerical targets whenever possible.
- Supporting Evidence from Data must explicitly cite at least THREE numerical values from the data above.
- Supporting Evidence from Data must contain at least 30 words.
- Supporting Evidence from Data must include:
  • at least one driver value,
  • at least one forest-loss metric,
  • at least one benchmark or emissions metric.
- Supporting Evidence from Data shorter than 30 words is invalid.
- Supporting Evidence from Data must include at least one value from DRIVER BREAKDOWN when available.
- At least one recommendation must compare the country against its regional benchmark.
- When regional benchmark data is available, explicitly reference the country's regional rank and regional average.
- Recommendations must explain whether the country performs above or below the regional average and what this implies.
- When interpreting regional rank, rank 1 means the highest forest loss in the region unless otherwise specified.
- Do not describe higher forest loss as better performance.
- Clearly distinguish between ranking position and performance quality.
- Prioritize recommendations according to the magnitude of forest-loss drivers.
- Compare the country against regional benchmarks when relevant.
- Use quantitative comparisons between drivers whenever possible.
- Use exact values from DRIVER BREAKDOWN whenever discussing drivers.
- Quantify the relative importance of drivers whenever possible.
- Explicitly compare Agriculture, Wildfire, Logging, Infrastructure, and other drivers using the values provided in DRIVER BREAKDOWN.
- Recommendations should prioritize the largest forest-loss driver unless evidence suggests otherwise.
- When discussing drivers, explain how much larger the primary driver is compared to the second and third largest drivers.
- Avoid generic statements such as "Agriculture is a major driver". Instead, use the exact values from DRIVER BREAKDOWN to justify priorities.
- If Prediction Confidence is 0:
  • do not use projected loss values,
  • do not use forecasted outcomes,
  • do not use future-loss estimates,
  • do not mention prediction accuracy improvements,
  • do not justify recommendations using projections.
- Use only observed historical data when Prediction Confidence is 0.
- Recommendations must address different strategic dimensions.
- Do not create multiple recommendations that solve the same problem in slightly different ways.
- Each recommendation must focus on a distinct intervention area.
- Recommendations 2-5 must not primarily address the same intervention area as Recommendation 1.

Possible intervention areas include:
- Monitoring and intelligence
- Policy and governance
- Community engagement
- Agricultural transition
- Supply-chain transformation
- Carbon markets
- Biodiversity protection
- Indigenous partnerships
- Restoration programs
- Research and innovation
- Regional cooperation
- Financing mechanisms

SUMMARY REQUIREMENTS:

- The summary must be 120-180 words.
- Include total historical loss.
- Include primary forest loss.
- Include CO2 emissions.
- Include regional rank.
- Include regional average forest loss.
- Identify the largest forest-loss driver using exact values.
- Explain how much larger the largest driver is compared to the second-largest driver.
- Explicitly compare total historical loss against the regional average and quantify the difference whenever possible.
- Explain what this means for the selected stakeholder.
- The summary should read like an executive briefing rather than a generic overview.
- The summary must explicitly identify:
  • the largest driver,
  • the second-largest driver,
  • how many times larger the largest driver is.
- The summary should read like an executive briefing prepared for senior decision-makers.
- Avoid generic statements.

- Recommendations must be actionable and implementable.
- Avoid recommendations that only suggest further study, evaluation, or discussion unless the stakeholder is academic_research.
- Every recommendation must propose a concrete intervention, program, policy, technology, financing mechanism, or operational action.

SUMMARY DIFFERENTIATION REQUIREMENTS:

The summary MUST be written from the perspective of the selected stakeholder and should read like a briefing prepared specifically for that audience.

- Corporate Sustainability:
  Focus on ESG risk, supply-chain exposure, supplier compliance, certification, sustainable sourcing, investor expectations, and business resilience.

- Policy & Governance:
  Focus on national performance, regulatory effectiveness, enforcement capacity, public investment, land-use planning, and institutional reforms.

- Academic & Research:
  Focus on research gaps, uncertainty reduction, monitoring improvements, methodological innovation, scientific evidence, and knowledge generation.

- Environmental NGO:
  Focus on community engagement, biodiversity protection, indigenous partnerships, conservation outcomes, environmental justice, and local capacity building.

Do not use priorities that belong primarily to another stakeholder group.

STAKEHOLDER-SPECIFIC IMPACT METRICS

The Expected Measurable Impact must reflect outcomes that the stakeholder can directly influence.

Policy & Governance:
Use metrics such as:
- regulatory compliance rates
- enforcement effectiveness
- protected area coverage
- public investment deployed
- adoption of national programs
- reduction in illegal activities
- land-use planning coverage

INVALID IMPACT EXAMPLES:
- Publish scientific papers
- Increase supplier certification
- Create NGO partnerships

VALID IMPACT EXAMPLES:
- 40% increase in regulatory compliance
- 30% reduction in illegal deforestation
- 20 million ha under monitoring
- 15% increase in enforcement coverage
- 25% increase in adoption of land-use plans

Avoid:
- supplier certification rates
- research publications
- NGO participation metrics

Academic & Research:
Use metrics such as:
- peer-reviewed publications
- model accuracy improvements
- monitoring system performance
- new datasets created
- methodological innovations
- research collaborations
- technology validation
- scientific adoption

INVALID IMPACT EXAMPLES:
- Reduce national deforestation by X%
- Increase regulatory compliance
- Improve supplier certification
- Increase ESG performance
- Reduce illegal logging by X%

VALID IMPACT EXAMPLES:
- Publish 20 peer-reviewed studies
- Improve model accuracy by 25%
- Create 3 national datasets
- Deploy 5 monitoring systems
- Establish 10 research collaborations
- Increase monitoring coverage by 40%
- Improve prediction accuracy by 30%
- Validate 4 new restoration methodologies

Avoid:
- direct reductions in national deforestation
- enforcement outcomes
- ESG ratings
- certification adoption

Environmental NGO:
Use metrics such as:
- communities engaged
- indigenous partnerships established
- hectares under community conservation
- participants trained
- biodiversity outcomes
- restoration activities completed
- grant-funded projects delivered

INVALID IMPACT EXAMPLES:
- Increase ESG ratings
- Pass national legislation
- Improve supplier compliance

VALID IMPACT EXAMPLES:
- 50 communities trained
- 20 indigenous partnerships established
- 100,000 ha under community conservation
- 10 biodiversity projects implemented
- 5,000 participants reached through awareness campaigns

Avoid:
- regulatory compliance rates
- national policy outcomes
- corporate procurement metrics

Corporate Sustainability:
Use metrics such as:
- supplier compliance rates
- traceability coverage
- certification adoption
- ESG performance
- sustainable sourcing coverage
- investment mobilized
- carbon credits generated
- supply-chain risk reduction

INVALID IMPACT EXAMPLES:
- Reduce national deforestation by 20%
- Increase regulatory enforcement
- Publish scientific studies

VALID IMPACT EXAMPLES:
- 80% supplier compliance coverage
- 70% traceability coverage
- 50% certified sourcing adoption
- 30% reduction in supply-chain risk exposure
- 25% increase in ESG score

Avoid:
- public enforcement outcomes
- scientific publications
- national legislation adoption

STAKEHOLDER-SPECIFIC OPENING LANGUAGE

Policy & Governance:
Start recommendations using:
- regulatory reform
- enforcement
- incentives
- public investment
- land-use governance
- national strategy

Academic & Research:
Start recommendations using:
- research program
- monitoring framework
- scientific study
- methodological innovation
- data platform
- field experiment

Environmental NGO:
Start recommendations using:
- community conservation
- indigenous partnerships
- training initiatives
- biodiversity protection
- awareness campaigns

Corporate Sustainability:
Start recommendations using:
- supply-chain transformation
- sustainable sourcing
- supplier engagement
- ESG risk management
- certification programs
- traceability systems

MANDATORY RECOMMENDATION FRAMING RULE

Recommendation objectives MUST begin with a stakeholder-specific intervention.

DO NOT begin objectives with:

- "Addressing the largest driver..."
- "Agriculture is the largest driver..."
- "Forest loss is significant..."
- Any generic data summary.

Instead:

Policy & Governance:
Begin with:
- Establish a national...
- Implement a regulatory...
- Create a public investment...
- Strengthen enforcement...

Academic & Research:
Begin with:
- Launch a research program...
- Establish a monitoring framework...
- Develop a scientific methodology...
- Create a national dataset...
- Conduct a multi-year field study...

Environmental NGO:
Begin with:
- Launch a community conservation initiative...
- Establish indigenous stewardship partnerships...
- Implement biodiversity protection programs...
- Develop community monitoring networks...

Corporate Sustainability:
Begin with:
- Implement a sustainable sourcing program...
- Deploy a supply-chain traceability system...
- Establish supplier compliance standards...
- Launch an ESG risk management initiative...
- Create a certification-based procurement program...

The objective must describe the intervention first.
Data should be used afterward to justify the intervention.

RECOMMENDATION ORDERING REQUIREMENTS:
TITLE REQUIREMENTS:

- Titles must be specific and action-oriented.
- Avoid generic titles such as:
  • Improve Monitoring
  • Strengthen Governance
  • Enhance Sustainability
  • Increase Awareness

- Titles should clearly describe the intervention.

Examples:
✓ National Agricultural Transition Program
✓ AI-Powered Deforestation Early Warning System
✓ Sustainable Supply-Chain Certification Initiative
✓ Indigenous Forest Stewardship Partnership Program
✓ Regional Forest Restoration Financing Facility

Recommendation 1:
- Must address the largest forest-loss driver.

Recommendation 2:

Policy & Governance:
- Monitoring and enforcement systems

Academic & Research:
- Monitoring methodologies and research infrastructure

Environmental NGO:
- Community monitoring and participatory surveillance

Corporate Sustainability:
- Supply-chain traceability and ESG monitoring

Recommendation 3:
Policy & Governance:
- Regulatory reform

Academic & Research:
- Research funding and innovation

Environmental NGO:
- Community conservation financing

Corporate Sustainability:
- ESG financing, sustainable procurement, supplier incentives

Examples by stakeholder:

policy_governance:
- governance
- enforcement
- institutions
- public financing

academic_research:
- research infrastructure
- data systems
- scientific collaboration
- research funding

environmental_ngo:
- program financing
- coalition building
- conservation governance
- implementation partnerships

corporate_sustainability:
- ESG governance
- supplier compliance
- certification systems
- sustainability reporting

Recommendation 4:
- Must focus on stakeholder-specific priorities.

If stakeholder = policy_governance:
- Recommendation 4 must focus on institutional reform, public financing, enforcement capacity, or government implementation.

If stakeholder = academic_research:
- Recommendation 4 must focus on research networks, methodological innovation, field studies, scientific collaboration, or data generation.

If stakeholder = environmental_ngo:
- Recommendation 4 must focus on community engagement, indigenous partnerships, conservation programs, capacity building, or biodiversity protection.

If stakeholder = corporate_sustainability:
- Recommendation 4 must focus on ESG implementation, supply-chain sustainability, certification systems, carbon markets, or corporate reporting.

Recommendation 5:
- Must focus on long-term resilience, restoration, regional competitiveness, climate adaptation, or future risk reduction.

DIVERSITY REQUIREMENTS:

- Each recommendation must address a different root cause, capability, or intervention mechanism.
- Recommendations must not repeat the same driver-focused solution from a different angle.
- If Recommendation 1 focuses on agricultural transition, Recommendations 2-5 must focus on different intervention areas.
- Avoid repeating agriculture-focused actions unless absolutely necessary.
- Each recommendation should create additional value beyond previous recommendations.

IMPORTANT:
- Recommendations must not overlap substantially.
- Each recommendation must solve a different problem.
- Repeating sustainable agriculture in multiple recommendations is not allowed.

IMPORTANT:

Do not place Implementation Timeframe,
Required Resources,
Expected Measurable Impact,
or Supporting Evidence inside the Objective section.

Do not place any section inside another section.

Each section must contain only its own content.

FORMATTING REQUIREMENTS:

Use actual line breaks.

After each section heading insert a newline.

Example:

**Objective**:
Text here.

**Specific Actions**:
1. Action one
2. Action two
3. Action three
4. Action four
5. Action five

**Implementation Timeframe**:
3-5 years

**Required Resources**:
Resources here.

**Expected Measurable Impact**:
Impact here.

**Supporting Evidence from Data**:
Evidence here.

Do not place all sections on a single line.

Each recommendation's 'text' field MUST follow this structure:

**Objective**: [50-100 words]

**Specific Actions**:
1. ...
2. ...
3. ...
4. ...
5. ...

**Implementation Timeframe**: [Time]

**Required Resources**: [Resources]

**Expected Measurable Impact**: [Impact with measurable outcomes]

**Supporting Evidence from Data**: [Reference actual data values]

CRITICAL OUTPUT REQUIREMENT:

You MUST return exactly 5 recommendations.

The recommendations array must contain:
- recommendation_number 1
- recommendation_number 2
- recommendation_number 3
- recommendation_number 4
- recommendation_number 5

Returning fewer than 5 recommendations is invalid.
Returning more than 5 recommendations is invalid.

Before producing the final JSON, verify that exactly 5 recommendations have been generated.

RETURN FORMAT (STRICT):

Return ONLY valid JSON. No explanation, no text before or after.

{{
  "summary": "string",
  "recommendations": [
    {{
      "recommendation_number": 1,
      "title": "string",
      "text": "string"
    }}
  ]
}}
"""
        return base_prompt

    def generate_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered recommendations using Structured Output"""
        try:
            prompt = self.generate_stakeholder_prompt(context)

            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.0,
            )

            # Parse the JSON response
            raw_text = response.output_text

            try:
                structured_data = json.loads(raw_text)

            except json.JSONDecodeError:
                import re
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)

                if match:
                    structured_data = json.loads(match.group())
                else:
                    raise ValueError("No valid JSON found in response")

            return {
                'success': True,
                'data': {
                    'country': context['country'],
                    'stakeholder': context['stakeholder'],
                    'generatedAt': context['generated_at'],
                    'summary': structured_data.get('summary'),
                    'recommendations': [
                        {
                            'id': r.get('recommendation_number'),
                            'title': r.get('title'),
                            'description': (
                                r.get('text')
                                if isinstance(r.get('text'), dict)
                                else {"raw_text": r.get('text')}
                            )
                        }
                        for r in structured_data.get('recommendations', [])
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {'success': False, 'error': str(e)}