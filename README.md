# Premier League AI Predictor

An intelligent football match prediction system that uses machine learning algorithms to predict Premier League match outcomes with statistical confidence scores.

## Features

- **Real-time Data Integration**: Fetches live Premier League data via football-data.org API
- **Advanced ML Models**: Uses Random Forest, Gradient Boosting, and Logistic Regression
- **Comprehensive Analysis**: Team statistics, head-to-head records, home/away performance
- **Interactive GUI**: User-friendly Tkinter interface with match selection and results visualization
- **Upcoming Fixtures**: Browse and predict upcoming Premier League matches
- **Historical Training**: Trains on multiple seasons of historical match data
- **Confidence Scoring**: Provides percentage confidence for each prediction
- **Multi-threaded Processing**: Non-blocking UI with background data processing

## Technologies Used

- **Programming Language**: Python 3.7+
- **GUI Framework**: Tkinter
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **API Integration**: requests
- **Data Source**: football-data.org API

## Usage

### Initial Setup
1. Launch the application
2. Enter your football-data.org API key
3. Click "Initialize" to fetch teams and set up the predictor
4. Wait for team data to load

### Making Predictions
1. Select home team from dropdown
2. Select away team from dropdown
3. Click "Predict Match Result"
4. Wait for model training (first time only)
5. View prediction results with confidence scores

### Using Fixtures
1. Click "Load Upcoming Fixtures"
2. Browse real upcoming Premier League matches
3. Double-click any fixture to auto-select teams
4. Generate instant predictions

## Machine Learning Models

The system employs three different algorithms and automatically selects the best performer:

### Models Implemented
- **Random Forest Classifier**: Ensemble learning using multiple decision trees
- **Gradient Boosting Classifier**: Sequential model building to minimize prediction errors
- **Logistic Regression**: Statistical baseline with feature scaling

### Features Analyzed
- Home/away win rates and goal statistics
- Overall team performance metrics
- Head-to-head historical records
- Home advantage factors
- Team strength differentials
- Attack vs defense matchups

### Model Performance
- **Overall Accuracy**: 55-65% (vs 33.3% random chance)
- **Training Data**: Last 3 seasons of Premier League matches
- **Validation Split**: 80% training, 20% testing

## Future Enhancements

- [ ] Player injury and suspension data integration
- [ ] Weather conditions analysis
- [ ] Multiple league support (La Liga, Serie A, Bundesliga)
- [ ] Web dashboard version
- [ ] Deep learning models (Neural Networks)
- [ ] Real-time match updates
- [ ] Database integration for faster data access

### Areas for Improvement
- Additional feature engineering
- Model optimization and new algorithms
- UI/UX enhancements
- Data visualization improvements
- Test coverage and documentation

## Acknowledgments

- [football-data.org](https://www.football-data.org/) for providing the Premier League API
- scikit-learn community for excellent machine learning tools
- Premier League for the exciting matches that make this project possible
