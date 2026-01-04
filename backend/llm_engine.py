import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
from typing import Dict, List, Any, Optional
import google.generativeai as genai
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
        """Initialize the recommendation engine with Gemini API and data"""
        genai.configure(api_key=api_key)
        # Using Gemini 1.5 Flash for speed and reliability with structured outputs
        self.model = genai.GenerativeModel('gemini-2.5-flash')
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

    def build_recommendation_context(self, country: str, stakeholder: str, data_range: Dict[str, int]) -> Dict[str, Any]:
        """Build comprehensive context for LLM recommendations"""
        historical_data = self.get_forest_data(country, data_range['startYear'], data_range['endYear'])
        predictions = self.get_predictions(country, years_ahead=5)
        drivers = self.analyze_deforestation_drivers(country)
        benchmarks = self.get_regional_benchmarks(country)
        
        return {
            'country': country,
            'stakeholder': stakeholder,
            'data_range': data_range,
            'historical_data': historical_data,
            'predictions': predictions,
            'drivers': drivers,
            'benchmarks': benchmarks,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def generate_stakeholder_prompt(self, context: Dict[str, Any]) -> str:
        """Generate stakeholder-specific prompt for LLM"""
        base_prompt = f"""
Generate 5-7 detailed, evidence-based recommendations for {context['stakeholder']} regarding {context['country']}'s forest loss.

DATA CONTEXT:
- Total Historical Loss: {context['historical_data'].get('total_loss', 0):,} ha
- Annual Trend: {context['historical_data'].get('trend_rate', 0):.1f}% ({context['historical_data'].get('trend_direction')})
- Primary Forest Loss: {context['historical_data'].get('primary_forest_loss', 0):,} ha
- CO2 Emissions: {context['historical_data'].get('carbon_emissions', 0):,.0f} Mg
- Drivers: {', '.join(context['drivers'].get('primary', []))}
- 5-Year Projection: {context['predictions'].get('projected_loss', 0):,} ha loss

CONSTRAINTS:
- No conversational filler or intros.
- Each recommendation's 'text' field MUST follow this structure:
  **Objective**: [Detailed goal]
  **Specific Actions**: [At least 3 specific actions]
  **Implementation Timeframe**: [Time]
  **Required Resources**: [Resources]
  **Expected Measurable Impact**: [Impact]
  **Supporting Evidence from Data**: [Explicitly mention the numbers from the data above]
"""
        return base_prompt

    def generate_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered recommendations using Structured Output"""
        try:
            prompt = self.generate_stakeholder_prompt(context)
            
            # Use structured output configuration
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": ForestAnalysisResponse,
                    "temperature": 0.0,
                }
            )
            
            # The SDK automatically parses the JSON into the Pydantic model
            # if using the correct version, otherwise use json.loads
            structured_data = json.loads(response.text)
            
            return {
                'success': True,
                'data': {
                    'country': context['country'],
                    'stakeholder': context['stakeholder'],
                    'generatedAt': context['generated_at'],
                    'summary': structured_data.get('summary'),
                    'recommendations': structured_data.get('recommendations')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {'success': False, 'error': str(e)}