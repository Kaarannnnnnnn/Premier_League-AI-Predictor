import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import threading
import time
import warnings
warnings.filterwarnings('ignore')

class PremierLeaguePredictor:
    def __init__(self, api_key=None):
        """Initialize the Premier League predictor with football-data.org API"""
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            'X-Auth-Token': api_key
        } if api_key else {}
        self.model = None
        self.scaler = None
        self.team_stats = {}
        self.head_to_head_stats = {}
        self.premier_league_id = 'PL'
        self.current_season = 2024
        self.teams_list = []
        self.best_model_name = ""
        
    def fetch_teams(self):
        """Fetch current Premier League teams"""
        if not self.api_key:
            return []
            
        try:
            response = requests.get(
                f"{self.base_url}/competitions/{self.premier_league_id}/teams",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                teams = [team['name'] for team in data['teams']]
                return sorted(teams)
            else:
                return []
                
        except requests.RequestException:
            return []
    
    def fetch_historical_matches(self, years_back=3):
        """Fetch historical match data for training (multiple seasons)"""
        if not self.api_key:
            return []
            
        all_matches = []
        current_year = datetime.now().year
        
        for year_offset in range(years_back):
            season_year = current_year - year_offset - 1
            
            try:
                response = requests.get(
                    f"{self.base_url}/competitions/{self.premier_league_id}/matches",
                    headers=self.headers,
                    params={
                        'season': season_year,
                        'status': 'FINISHED'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'matches' in data and len(data['matches']) > 0:
                        season_matches = self._convert_api_matches(data['matches'])
                        all_matches.extend(season_matches)
                
                time.sleep(6)  # Rate limiting
                
            except requests.RequestException:
                continue
        
        return all_matches
    
    def _convert_api_matches(self, api_matches):
        """Convert football-data.org API matches to internal format"""
        converted_matches = []
        
        for match in api_matches:
            try:
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
                if match['status'] != 'FINISHED' or not match['score']['fullTime']:
                    continue
                    
                home_goals = match['score']['fullTime']['home']
                away_goals = match['score']['fullTime']['away']
                
                if home_goals is None or away_goals is None:
                    continue
                
                if home_goals > away_goals:
                    winner = 'HOME_TEAM'
                elif away_goals > home_goals:
                    winner = 'AWAY_TEAM'
                else:
                    winner = 'DRAW'
                
                converted_match = {
                    'homeTeam': {'name': home_team},
                    'awayTeam': {'name': away_team},
                    'score': {
                        'fullTime': {
                            'home': home_goals,
                            'away': away_goals
                        }
                    },
                    'winner': winner,
                    'status': 'FINISHED',
                    'utcDate': match['utcDate']
                }
                
                converted_matches.append(converted_match)
                
            except KeyError:
                continue
        
        return converted_matches
    
    def get_current_fixtures(self):
        """Fetch upcoming Premier League fixtures"""
        if not self.api_key:
            return []
            
        try:
            response = requests.get(
                f"{self.base_url}/competitions/{self.premier_league_id}/matches",
                headers=self.headers,
                params={'status': 'SCHEDULED'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'matches' in data and len(data['matches']) > 0:
                    fixtures = []
                    for fixture in data['matches'][:15]:
                        fixtures.append({
                            'home_team': fixture['homeTeam']['name'],
                            'away_team': fixture['awayTeam']['name'],
                            'date': fixture['utcDate'][:10],
                            'matchday': fixture.get('matchday', 'N/A')
                        })
                    return fixtures
            return []
                
        except requests.RequestException:
            return []
    
    def calculate_team_stats(self, matches):
        """Calculate comprehensive team statistics"""
        team_stats = {}
        
        for match in matches:
            if match['status'] != 'FINISHED':
                continue
                
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            winner = match.get('winner', 'DRAW')
            
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'home_wins': 0, 'home_draws': 0, 'home_losses': 0, 'home_games': 0,
                        'away_wins': 0, 'away_draws': 0, 'away_losses': 0, 'away_games': 0,
                        'total_wins': 0, 'total_draws': 0, 'total_losses': 0, 'total_games': 0,
                        'goals_scored_home': 0, 'goals_conceded_home': 0,
                        'goals_scored_away': 0, 'goals_conceded_away': 0,
                        'points': 0
                    }
            
            # Update home team stats
            team_stats[home_team]['home_games'] += 1
            team_stats[home_team]['total_games'] += 1
            team_stats[home_team]['goals_scored_home'] += match['score']['fullTime']['home']
            team_stats[home_team]['goals_conceded_home'] += match['score']['fullTime']['away']
            
            if winner == 'HOME_TEAM':
                team_stats[home_team]['home_wins'] += 1
                team_stats[home_team]['total_wins'] += 1
                team_stats[home_team]['points'] += 3
            elif winner == 'DRAW':
                team_stats[home_team]['home_draws'] += 1
                team_stats[home_team]['total_draws'] += 1
                team_stats[home_team]['points'] += 1
            else:
                team_stats[home_team]['home_losses'] += 1
                team_stats[home_team]['total_losses'] += 1
            
            # Update away team stats
            team_stats[away_team]['away_games'] += 1
            team_stats[away_team]['total_games'] += 1
            team_stats[away_team]['goals_scored_away'] += match['score']['fullTime']['away']
            team_stats[away_team]['goals_conceded_away'] += match['score']['fullTime']['home']
            
            if winner == 'AWAY_TEAM':
                team_stats[away_team]['away_wins'] += 1
                team_stats[away_team]['total_wins'] += 1
                team_stats[away_team]['points'] += 3
            elif winner == 'DRAW':
                team_stats[away_team]['away_draws'] += 1
                team_stats[away_team]['total_draws'] += 1
                team_stats[away_team]['points'] += 1
            else:
                team_stats[away_team]['away_losses'] += 1
                team_stats[away_team]['total_losses'] += 1
        
        return team_stats
    
    def calculate_head_to_head(self, matches):
        """Calculate head-to-head statistics"""
        h2h_stats = {}
        
        for match in matches:
            if match['status'] != 'FINISHED':
                continue
                
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            winner = match.get('winner', 'DRAW')
            
            teams = tuple(sorted([home_team, away_team]))
            
            if teams not in h2h_stats:
                h2h_stats[teams] = {
                    'total_matches': 0,
                    f'{teams[0]}_wins': 0,
                    f'{teams[1]}_wins': 0,
                    'draws': 0,
                    'goals_for_team1': 0,
                    'goals_for_team2': 0
                }
            
            h2h_stats[teams]['total_matches'] += 1
            
            if home_team == teams[0]:
                h2h_stats[teams]['goals_for_team1'] += match['score']['fullTime']['home']
                h2h_stats[teams]['goals_for_team2'] += match['score']['fullTime']['away']
            else:
                h2h_stats[teams]['goals_for_team1'] += match['score']['fullTime']['away']
                h2h_stats[teams]['goals_for_team2'] += match['score']['fullTime']['home']
            
            if winner == 'HOME_TEAM':
                h2h_stats[teams][f'{home_team}_wins'] += 1
            elif winner == 'AWAY_TEAM':
                h2h_stats[teams][f'{away_team}_wins'] += 1
            else:
                h2h_stats[teams]['draws'] += 1
        
        return h2h_stats
    
    def create_features(self, home_team, away_team, team_stats, h2h_stats):
        """Create comprehensive features for prediction"""
        features = {}
        
        home_stats = team_stats.get(home_team, {})
        if home_stats.get('home_games', 0) > 0:
            features['home_win_rate_home'] = home_stats.get('home_wins', 0) / home_stats['home_games']
            features['home_goals_per_game_home'] = home_stats.get('goals_scored_home', 0) / home_stats['home_games']
            features['home_goals_conceded_per_game_home'] = home_stats.get('goals_conceded_home', 0) / home_stats['home_games']
        else:
            features['home_win_rate_home'] = 0.45
            features['home_goals_per_game_home'] = 1.3
            features['home_goals_conceded_per_game_home'] = 1.0
        
        away_stats = team_stats.get(away_team, {})
        if away_stats.get('away_games', 0) > 0:
            features['away_win_rate_away'] = away_stats.get('away_wins', 0) / away_stats['away_games']
            features['away_goals_per_game_away'] = away_stats.get('goals_scored_away', 0) / away_stats['away_games']
            features['away_goals_conceded_per_game_away'] = away_stats.get('goals_conceded_away', 0) / away_stats['away_games']
        else:
            features['away_win_rate_away'] = 0.25
            features['away_goals_per_game_away'] = 1.0
            features['away_goals_conceded_per_game_away'] = 1.3
        
        if home_stats.get('total_games', 0) > 0:
            features['home_overall_win_rate'] = home_stats.get('total_wins', 0) / home_stats['total_games']
            features['home_points_per_game'] = home_stats.get('points', 0) / home_stats['total_games']
        else:
            features['home_overall_win_rate'] = 0.35
            features['home_points_per_game'] = 1.2
            
        if away_stats.get('total_games', 0) > 0:
            features['away_overall_win_rate'] = away_stats.get('total_wins', 0) / away_stats['total_games']
            features['away_points_per_game'] = away_stats.get('points', 0) / away_stats['total_games']
        else:
            features['away_overall_win_rate'] = 0.35
            features['away_points_per_game'] = 1.2
        
        teams = tuple(sorted([home_team, away_team]))
        h2h = h2h_stats.get(teams, {})
        
        if h2h.get('total_matches', 0) > 0:
            features['h2h_home_win_rate'] = h2h.get(f'{home_team}_wins', 0) / h2h['total_matches']
            features['h2h_away_win_rate'] = h2h.get(f'{away_team}_wins', 0) / h2h['total_matches']
            features['h2h_draw_rate'] = h2h.get('draws', 0) / h2h['total_matches']
        else:
            features['h2h_home_win_rate'] = 0.4
            features['h2h_away_win_rate'] = 0.3
            features['h2h_draw_rate'] = 0.3
        
        features['home_advantage'] = 1.0
        features['strength_difference'] = features['home_points_per_game'] - features['away_points_per_game']
        features['attack_vs_defense'] = features['home_goals_per_game_home'] - features['away_goals_conceded_per_game_away']
        
        return list(features.values())
    
    def train_model(self, years_back=3, progress_callback=None):
        """Train the prediction model with real data"""
        if not self.api_key:
            return False, "No API key provided"
        
        if progress_callback:
            progress_callback("Fetching historical match data...")
        
        all_matches = self.fetch_historical_matches(years_back)
        
        if len(all_matches) < 100:
            return False, f"Insufficient data: only {len(all_matches)} matches found"
        
        if progress_callback:
            progress_callback("Calculating team statistics...")
        
        self.team_stats = self.calculate_team_stats(all_matches)
        self.head_to_head_stats = self.calculate_head_to_head(all_matches)
        
        if progress_callback:
            progress_callback("Creating training features...")
        
        X, y = [], []
        
        for match in all_matches:
            if match['status'] != 'FINISHED':
                continue
                
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            winner = match.get('winner', 'DRAW')
            
            features = self.create_features(home_team, away_team, self.team_stats, self.head_to_head_stats)
            X.append(features)
            y.append(winner)
        
        if len(X) < 100:
            return False, f"Insufficient training samples: {len(X)}"
        
        X = np.array(X)
        y = np.array(y)
        
        if progress_callback:
            progress_callback(f"Training on {len(X)} matches...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            if progress_callback:
                progress_callback(f"Training {name} model...")
                
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            score = accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.model = best_model
        self.best_model_name = best_name
        
        return True, f"Model trained successfully! Best: {best_name} ({best_score*100:.2f}% accuracy)"
    
    def predict_match(self, home_team, away_team):
        """Predict match outcome"""
        if self.model is None:
            return {"error": "Model not trained yet"}
        
        if home_team not in self.teams_list or away_team not in self.teams_list:
            return {"error": "Invalid team name(s)"}
        
        features = self.create_features(home_team, away_team, self.team_stats, self.head_to_head_stats)
        features = np.array(features).reshape(1, -1)
        
        if self.best_model_name == 'LogisticRegression':
            features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        
        prob_dict = {}
        for i, class_name in enumerate(classes):
            if class_name == 'HOME_TEAM':
                prob_dict[f'{home_team} Win'] = probabilities[i] * 100
            elif class_name == 'AWAY_TEAM':
                prob_dict[f'{away_team} Win'] = probabilities[i] * 100
            else:
                prob_dict['Draw'] = probabilities[i] * 100
        
        if prediction == 'HOME_TEAM':
            winner = home_team
        elif prediction == 'AWAY_TEAM':
            winner = away_team
        else:
            winner = 'Draw'
        
        confidence = max(probabilities) * 100
        
        return {
            "winner": winner,
            "probabilities": prob_dict,
            "confidence": confidence,
            "model_used": self.best_model_name,
            "home_team": home_team,
            "away_team": away_team
        }


class PremierLeagueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 Premier League Match Predictor")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Make window resizable
        self.root.resizable(True, True)
        
        self.predictor = None
        self.model_trained = False
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='green', background='#f0f0f0')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='red', background='#f0f0f0')
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏆 Premier League Match Predictor", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # API Section
        api_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding=20)
        api_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(api_frame, text="Get your FREE API key from: https://www.football-data.org/client/register").pack(anchor='w')
        
        api_input_frame = ttk.Frame(api_frame)
        api_input_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(api_input_frame, text="API Key:").pack(side='left')
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(api_input_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_entry.pack(side='left', padx=(10, 10), fill='x', expand=True)
        
        self.init_button = ttk.Button(api_input_frame, text="Initialize", command=self.initialize_predictor)
        self.init_button.pack(side='right')
        
        self.api_status_label = ttk.Label(api_frame, text="")
        self.api_status_label.pack(anchor='w', pady=(10, 0))
        
        # Main Interface (initially hidden)
        self.main_interface = ttk.Frame(main_frame)
        
        # Team Selection
        team_frame = ttk.LabelFrame(self.main_interface, text="Team Selection", padding=20)
        team_frame.pack(fill='x', pady=(0, 20))
        
        # Team selection grid
        selection_grid = ttk.Frame(team_frame)
        selection_grid.pack(fill='x')
        
        # Configure grid columns
        selection_grid.columnconfigure(0, weight=1)
        selection_grid.columnconfigure(1, weight=0)
        selection_grid.columnconfigure(2, weight=1)
        
        # Home team
        home_frame = ttk.Frame(selection_grid)
        home_frame.grid(row=0, column=0, padx=(0, 20), sticky='ew')
        ttk.Label(home_frame, text="🏠 Home Team:", style='Header.TLabel').pack(anchor='w')
        self.home_team_var = tk.StringVar()
        self.home_combo = ttk.Combobox(home_frame, textvariable=self.home_team_var, state="readonly", width=25)
        self.home_combo.pack(fill='x', pady=(5, 0))
        
        # VS badge
        vs_label = ttk.Label(selection_grid, text="VS", font=('Arial', 14, 'bold'))
        vs_label.grid(row=0, column=1, padx=20)
        
        # Away team
        away_frame = ttk.Frame(selection_grid)
        away_frame.grid(row=0, column=2, padx=(20, 0), sticky='ew')
        ttk.Label(away_frame, text="🛣️ Away Team:", style='Header.TLabel').pack(anchor='w')
        self.away_team_var = tk.StringVar()
        self.away_combo = ttk.Combobox(away_frame, textvariable=self.away_team_var, state="readonly", width=25)
        self.away_combo.pack(fill='x', pady=(5, 0))
        
        # Predict button
        button_frame = ttk.Frame(team_frame)
        button_frame.pack(pady=(20, 0))
        self.predict_button = ttk.Button(button_frame, text="🔮 Predict Match Result", command=self.predict_match)
        self.predict_button.pack()
        
        # Progress bar and status
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(team_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(team_frame, mode='indeterminate')
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(self.main_interface, text="Prediction Results", padding=20)
        
        # Results content (will be populated dynamically)
        self.results_content = ttk.Frame(self.results_frame)
        self.results_content.pack(fill='both', expand=True)
        
        # Fixtures Section
        fixtures_frame = ttk.LabelFrame(self.main_interface, text="Upcoming Fixtures", padding=20)
        fixtures_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Fixtures button
        fixtures_button_frame = ttk.Frame(fixtures_frame)
        fixtures_button_frame.pack(fill='x')
        ttk.Button(fixtures_button_frame, text="📅 Load Upcoming Fixtures", command=self.load_fixtures).pack(side='left')
        
        # Fixtures list
        fixtures_list_frame = ttk.Frame(fixtures_frame)
        fixtures_list_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Create treeview for fixtures
        columns = ('Home Team', 'Away Team', 'Date')
        self.fixtures_tree = ttk.Treeview(fixtures_list_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.fixtures_tree.heading(col, text=col)
            self.fixtures_tree.column(col, width=120)
        
        # Scrollbar for fixtures
        fixtures_scrollbar = ttk.Scrollbar(fixtures_list_frame, orient='vertical', command=self.fixtures_tree.yview)
        self.fixtures_tree.configure(yscrollcommand=fixtures_scrollbar.set)
        
        self.fixtures_tree.pack(side='left', fill='both', expand=True)
        fixtures_scrollbar.pack(side='right', fill='y')
        
        # Bind double-click to fixtures
        self.fixtures_tree.bind('<Double-1>', self.on_fixture_double_click)
        
    def initialize_predictor(self):
        """Initialize the predictor with API key"""
        api_key = self.api_key_var.get().strip()
        
        if not api_key:
            self.show_status("Please enter your API key", "error")
            return
        
        self.init_button.config(state='disabled', text="Initializing...")
        self.progress_bar.pack(pady=(5, 0))
        self.progress_bar.start()
        
        def init_thread():
            try:
                self.predictor = PremierLeaguePredictor(api_key=api_key)
                self.predictor.teams_list = self.predictor.fetch_teams()
                
                if not self.predictor.teams_list:
                    self.root.after(0, lambda: self.show_status("Failed to fetch teams. Check your API key.", "error"))
                    return
                
                # Populate team dropdowns
                self.root.after(0, self.populate_teams)
                self.root.after(0, lambda: self.show_status("✅ Predictor initialized successfully!", "success"))
                self.root.after(0, self.show_main_interface)
                
            except Exception as e:
                self.root.after(0, lambda: self.show_status(f"Error: {str(e)}", "error"))
            finally:
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def populate_teams(self):
        """Populate team comboboxes"""
        teams = self.predictor.teams_list
        self.home_combo['values'] = teams
        self.away_combo['values'] = teams
    
    def show_main_interface(self):
        """Show the main prediction interface"""
        self.main_interface.pack(fill='both', expand=True, pady=(20, 0))
    
    def show_status(self, message, status_type="info"):
        """Show status message"""
        if status_type == "success":
            self.api_status_label.configure(text=message, style='Success.TLabel')
        elif status_type == "error":
            self.api_status_label.configure(text=message, style='Error.TLabel')
        else:
            self.api_status_label.configure(text=message)
    
    def stop_progress(self):
        """Stop progress bar and reset button"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.init_button.config(state='normal', text="Initialize")
        self.progress_var.set("")
    
    def predict_match(self):
        """Predict match outcome"""
        home_team = self.home_team_var.get()
        away_team = self.away_team_var.get()
        
        if not home_team or not away_team:
            messagebox.showwarning("Warning", "Please select both teams")
            return
        
        if home_team == away_team:
            messagebox.showwarning("Warning", "Please select different teams")
            return
        
        if not self.model_trained:
            self.train_and_predict(home_team, away_team)
        else:
            self.make_prediction(home_team, away_team)
    
    def train_and_predict(self, home_team, away_team):
        """Train model and make prediction"""
        self.predict_button.config(state='disabled', text="Training Model...")
        self.progress_bar.pack(pady=(5, 0))
        self.progress_bar.start()
        
        def training_thread():
            try:
                def progress_callback(message):
                    self.root.after(0, lambda msg=message: self.progress_var.set(msg))
                
                # Train the model
                success, message = self.predictor.train_model(years_back=3, progress_callback=progress_callback)
                
                if success:
                    self.model_trained = True
                    self.root.after(0, lambda: self.progress_var.set("Model trained! Making prediction..."))
                    
                    # Make prediction
                    result = self.predictor.predict_match(home_team, away_team)
                    
                    # Display results on main thread
                    self.root.after(0, lambda res=result: self.display_prediction(res))
                    
                else:
                    self.root.after(0, lambda msg=message: messagebox.showerror("Training Failed", msg))
                    
            except Exception as e:
                error_msg = f"Training failed: {str(e)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
            finally:
                self.root.after(0, self.stop_prediction_progress)
        
        threading.Thread(target=training_thread, daemon=True).start()
    
    def make_prediction(self, home_team, away_team):
        """Make prediction with trained model"""
        self.predict_button.config(state='disabled', text="Predicting...")
        
        def prediction_thread():
            try:
                result = self.predictor.predict_match(home_team, away_team)
                # Display results on main thread
                self.root.after(0, lambda res=result: self.display_prediction(res))
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
            finally:
                self.root.after(0, lambda: self.predict_button.config(state='normal', text="🔮 Predict Match Result"))
        
        threading.Thread(target=prediction_thread, daemon=True).start()
    
    def display_prediction(self, result):
        """Display prediction results"""
        if "error" in result:
            messagebox.showerror("Error", result["error"])
            return
        
        # Clear previous results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Ensure results frame is visible
        self.results_frame.pack(fill='x', pady=(20, 0))
        
        # Create a scrollable frame for results
        canvas = tk.Canvas(self.results_content, height=300)
        scrollbar = ttk.Scrollbar(self.results_content, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Winner announcement section
        winner_section = ttk.LabelFrame(scrollable_frame, text="🏆 Match Prediction", padding=15)
        winner_section.pack(fill='x', pady=(0, 15))
        
        # Match info
        match_info = ttk.Label(winner_section, 
                              text=f"{result['home_team']} vs {result['away_team']}", 
                              font=('Arial', 12, 'bold'))
        match_info.pack(pady=(0, 10))
        
        # Winner announcement
        winner_label = ttk.Label(winner_section, 
                                text=f"Predicted Winner: {result['winner']}", 
                                font=('Arial', 16, 'bold'), 
                                foreground='darkgreen')
        winner_label.pack(pady=(0, 5))
        
        # Confidence
        confidence_label = ttk.Label(winner_section, 
                                   text=f"Confidence: {result['confidence']:.1f}%", 
                                   font=('Arial', 12))
        confidence_label.pack(pady=(0, 5))
        
        # Model info
        model_label = ttk.Label(winner_section, 
                              text=f"AI Model: {result['model_used']}", 
                              font=('Arial', 10), 
                              foreground='gray')
        model_label.pack()
        
        # Probabilities section
        prob_section = ttk.LabelFrame(scrollable_frame, text="📊 Match Probabilities", padding=15)
        prob_section.pack(fill='x', pady=(0, 15))
        
        # Create probability display
        for outcome, prob in result['probabilities'].items():
            outcome_frame = ttk.Frame(prob_section)
            outcome_frame.pack(fill='x', pady=8)
            
            # Outcome name
            outcome_label = ttk.Label(outcome_frame, text=f"{outcome}:", 
                                    font=('Arial', 11, 'bold'), width=25, anchor='w')
            outcome_label.pack(side='left')
            
            # Progress bar for visual representation
            prob_bar_frame = ttk.Frame(outcome_frame)
            prob_bar_frame.pack(side='left', fill='x', expand=True, padx=(10, 10))
            
            prob_bar = ttk.Progressbar(prob_bar_frame, length=200, mode='determinate', value=prob)
            prob_bar.pack(side='left')
            
            # Percentage
            percent_label = ttk.Label(outcome_frame, text=f"{prob:.1f}%", 
                                    font=('Arial', 11, 'bold'), width=8, anchor='e',
                                    foreground='darkblue')
            percent_label.pack(side='right')
        
        # Additional info section
        info_section = ttk.LabelFrame(scrollable_frame, text="ℹ️ Additional Information", padding=15)
        info_section.pack(fill='x')
        
        info_text = f"""
• This prediction is based on historical Premier League match data
• The AI model analyzes team performance, head-to-head records, and form
• Higher confidence indicates stronger statistical evidence for the prediction
• Results are for entertainment purposes - actual matches may vary!
        """
        
        info_label = ttk.Label(info_section, text=info_text.strip(), 
                             font=('Arial', 9), justify='left')
        info_label.pack(anchor='w')
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Force update of the display
        self.root.update_idletasks()
        
        # Show success message
        messagebox.showinfo("Prediction Complete", 
                          f"Match prediction ready!\n\n"
                          f"Predicted Winner: {result['winner']}\n"
                          f"Confidence: {result['confidence']:.1f}%")
    
    def stop_prediction_progress(self):
        """Stop prediction progress indicators"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.predict_button.config(state='normal', text="🔮 Predict Match Result")
        self.progress_var.set("")
    
    def load_fixtures(self):
        """Load upcoming fixtures"""
        if not self.predictor:
            messagebox.showwarning("Warning", "Please initialize the predictor first")
            return
        
        def fixtures_thread():
            try:
                fixtures = self.predictor.get_current_fixtures()
                self.root.after(0, lambda: self.display_fixtures(fixtures))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load fixtures: {str(e)}"))
        
        threading.Thread(target=fixtures_thread, daemon=True).start()
    
    def display_fixtures(self, fixtures):
        """Display fixtures in the treeview"""
        # Clear existing items
        for item in self.fixtures_tree.get_children():
            self.fixtures_tree.delete(item)
        
        if not fixtures:
            messagebox.showinfo("Info", "No upcoming fixtures found")
            return
        
        # Add fixtures to treeview
        for fixture in fixtures:
            self.fixtures_tree.insert('', 'end', values=(
                fixture['home_team'],
                fixture['away_team'],
                fixture['date']
            ))
    
    def on_fixture_double_click(self, event):
        """Handle double-click on fixture"""
        selection = self.fixtures_tree.selection()
        if not selection:
            return
        
        item = self.fixtures_tree.item(selection[0])
        values = item['values']
        
        if len(values) >= 2:
            home_team = values[0]
            away_team = values[1]
            
            # Set the teams in dropdowns
            self.home_team_var.set(home_team)
            self.away_team_var.set(away_team)
            
            # Ask if user wants to predict
            if messagebox.askyesno("Predict Match", 
                                 f"Do you want to predict {home_team} vs {away_team}?"):
                self.predict_match()


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set icon if available (optional)
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('premier_league_icon.ico')
        pass
    except:
        pass
    
    app = PremierLeagueGUI(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
