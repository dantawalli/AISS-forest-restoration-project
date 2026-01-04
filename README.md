# 🌳 Forest Watch Project

A comprehensive forest monitoring dashboard with AI-powered recommendations and predictive analytics.

## 📊 Overview

Forest Watch provides real-time monitoring and analysis of global deforestation trends using satellite data, machine learning predictions, and AI-powered recommendations tailored for different stakeholder groups.

## 🚀 Features

### 📈 Data Analytics
- **Historical Analysis**: 2001-2024 forest loss trends
- **Country Comparisons**: Regional benchmarks and rankings
- **Driver Analysis**: Deforestation causes (agriculture, logging, wildfire)
- **Carbon Emissions**: Environmental impact tracking

### 🤖 AI-Powered Recommendations
- **Stakeholder-Specific**: Tailored insights for:
  - Policy & Governance
  - Academic Research
  - Environmental NGOs
  - Corporate Sustainability
- **Evidence-Based**: All recommendations grounded in real data
- **Structured Output**: Clear objectives, actions, timelines, and impact metrics

### 🔮 Predictive Analytics
- **ML Predictions**: 2025-2035 forest loss projections
- **Risk Assessment**: Confidence scores and risk areas
- **Multi-Country Support**: Batch predictions for comparative analysis

## 🏗️ Architecture

```
Frontend (React) → Backend (Flask API) → Database (PostgreSQL)
                      ↓
                LLM Engine (Gemini AI) → CSV Predictions
                ↓
            ML Models (XGBoost) → Data Processing
```

## 📡 API Endpoints

### 🌍 Data Endpoints
- `GET /api/summary` - Global forest statistics
- `GET /api/countries` - Available countries list
- `GET /api/countries/search?q=<query>` - Fuzzy country search
- `GET /api/country-info?q=<query>` - Country code resolution
- `GET /api/loss-trend?country=<name>` - Historical loss trends
- `GET /api/drivers?country=<name>` - Deforestation drivers
- `GET /api/emissions?country=<name>` - Carbon emissions data
- `GET /api/map-data?year=<year>` - Geospatial mapping data

### 🔮 Prediction Endpoints
- `POST /api/predict` - Forest loss predictions
  ```json
  // Single country
  {
    "country": "Australia",
    "year": 2030
  }
  
  // Multiple countries
  {
    "countries": ["Australia", "Brazil", "Indonesia"],
    "year": 2030
  }
  ```

### 🤖 AI Recommendation Endpoints
- `POST /api/recommendations` - AI-powered recommendations
  ```json
  {
    "country": "Australia",
    "stakeholder": "policy_governance",
    "dataRange": {"startYear": 2001, "endYear": 2024},
    "includePredictions": true,
    "language": "en"
  }
  ```
- `POST /api/insights` - Deep analytical insights
- `GET /api/recommendations/templates` - Available stakeholder types

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask (Python)
- **Database**: PostgreSQL with pandas data processing
- **ML/AI**: 
  - XGBoost for predictions
  - Google Gemini 2.5 Flash for recommendations
- **Data Processing**: NumPy, pandas, scikit-learn

### Frontend
- **Framework**: React (assumed)
- **Visualization**: Chart.js, D3.js (recommended)
- **Maps**: Leaflet, Mapbox (recommended)

### Data Sources
- **Primary**: `data/processed/merged_clean_data.csv`
- **Predictions**: `data/predicted_tree_cover_loss_2025_2035.csv`
- **Models**: `models/tree_cover_loss_model.pkl`

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, defaults to CSV)
- Google Gemini API key (for recommendations)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd forest-watch-project

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### Running the Application

```bash
# Start backend server
cd backend
python app.py

# Server runs on http://127.0.0.1:5001
```

### Environment Variables

```bash
# Required for AI recommendations
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379

# Optional: Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/forest_watch
```

## 📊 Data Structure

### Historical Data Format
```csv
country,year,tree_cover_loss_ha,primary_forest_loss_ha,carbon_gross_emissions_MgCO2e,hard_commodities,logging,wildfire,...
Australia,2001,123456.78,98765.43,1234567.89,123.45,67.89,12.34,...
```

### Prediction Data Format
```csv
country,year,predicted_loss
Australia,2025,666486.17
Australia,2026,687948.34
Australia,2027,709410.50
```

## 🌍 Country Support

### ✅ Supported Countries
167 countries with complete historical data (2001-2024)

### 🔍 Country Resolution
- **Full Names**: "Australia", "United States", "Brazil"
- **Country Codes**: "AUS" → "Australia", "BRA" → "Brazil"
- **Fuzzy Search**: Partial matching for country names

### 📊 Regional Groupings
- **South America**: Brazil, Argentina, Colombia, Peru...
- **Southeast Asia**: Indonesia, Malaysia, Thailand...
- **Africa**: DRC, Nigeria, Cameroon...
- **Europe**: Sweden, Finland, France...
- **North America**: Canada, United States, Mexico...

## 🤖 AI Recommendations

### Stakeholder Types

#### 🏛️ Policy & Governance
- Legislative actions and enforcement mechanisms
- International cooperation opportunities
- Budget allocation and monitoring frameworks

#### 🎓 Academic Research
- Research priorities and methodology improvements
- Collaboration opportunities and technology needs
- Policy-relevant research translation

#### 🌱 Environmental NGOs
- Advocacy campaigns and conservation interventions
- Community engagement and partnerships
- Monitoring and accountability mechanisms

#### 🏢 Corporate Sustainability
- Supply chain sustainability and ESG reporting
- Stakeholder engagement and investment opportunities
- Risk management and mitigation strategies

### Recommendation Format
```json
{
  "executiveSummary": "Australia faces accelerating deforestation...",
  "recommendations": [
    {
      "recommendation_number": 1,
      "title": "Strengthen Forest Protection Laws",
      "objective": "Reduce illegal logging by 40%",
      "specific_actions": [
        "Increase satellite monitoring coverage",
        "Implement real-time alert system",
        "Establish rapid response units"
      ],
      "implementation_timeframe": "24-36 months",
      "required_resources": "$15.2M for equipment, 200 personnel",
      "expected_measurable_impact": "40% reduction in illegal logging",
      "supporting_evidence_from_data": "15% annual increase in forest loss, primarily from logging"
    }
  ]
}
```

## 📈 Performance Features

### 🚀 Speed Optimizations
- **CSV-based predictions**: Faster than real-time ML inference
- **Data caching**: Redis support for frequent queries
- **Batch processing**: Multi-country predictions in single request
- **JSON serialization**: Optimized for NumPy/Pandas data types

### 🔒 Security & Rate Limiting
- **Rate Limits**: 100 requests/minute per IP
- **Input Validation**: Sanitized country names and parameters
- **Error Handling**: Structured JSON error responses
- **CORS**: Enabled for frontend integration

## 🌐 Deployment

### Development
```bash
# Local development
python app.py
# Runs on http://127.0.0.1:5001
```

### Production Hosting

#### Recommended Stack
- **Frontend**: Vercel (React deployment)
- **Backend**: AWS ECS/Railway/Render
- **Database**: AWS RDS PostgreSQL
- **AI/ML**: AWS SageMaker (optional)

#### Environment Setup
```bash
# Production environment variables
export FLASK_ENV=production
export GEMINI_API_KEY=prod_key_here
export DATABASE_URL=prod_database_url
```

## 🧪 Testing

### API Testing
```bash
# Test data endpoints
curl http://127.0.0.1:5001/api/countries

# Test predictions
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "Australia", "year": 2030}'

# Test recommendations
curl -X POST http://127.0.0.1:5001/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"country": "Australia", "stakeholder": "policy_governance", "dataRange": {"startYear": 2001, "endYear": 2024}}'
```

## 🔧 Configuration

### Model Settings
- **Prediction Years**: 2025-2035 (10-year forecast)
- **Confidence Threshold**: 75% base confidence
- **Risk Areas**: Automatic detection based on trends
- **Data Threshold**: Tree cover loss at 30% canopy density

### LLM Settings
- **Model**: Gemini 2.5 Flash
- **Temperature**: 0.0 (deterministic output)
- **Response Format**: Structured JSON with Pydantic validation
- **Max Recommendations**: 5-7 per request

## 🐛 Troubleshooting

### Common Issues

#### Country Not Found
```bash
# Check available countries
curl http://127.0.0.1:5001/api/countries

# Resolve country codes
curl http://127.0.0.1:5001/api/country-info?q=AUS
```

#### LLM Recommendations Not Working
```bash
# Check API key
echo $GEMINI_API_KEY

# Verify predictions file exists
ls data/predicted_tree_cover_loss_2025_2035.csv
```

#### Performance Issues
- **Enable Redis caching** for frequent queries
- **Use batch predictions** for multiple countries
- **Check CSV file size** (should be < 100MB)

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python app.py
```

### Code Standards
- **Python**: PEP 8 compliant
- **Documentation**: Docstrings for all functions
- **Error Handling**: Structured JSON responses
- **Logging**: Comprehensive error tracking

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Create GitHub issue
- **Features**: Submit enhancement request
- **Security**: Report vulnerabilities privately

---

**Built with ❤️ for global forest conservation** 🌍🌳