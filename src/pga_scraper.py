"""
PGA Tour Statistics Scraper
Fetches current season player statistics from pgatour.com
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time
import json


class PGATourScraper:
    """Scrape current PGA Tour statistics for player-course fit modeling."""

    # PGA Tour stats API endpoints (discovered from their website)
    STATS_API_BASE = "https://www.pgatour.com/api/stats"

    # Key stat IDs for our model
    STAT_IDS = {
        'sg_total': '02675',
        'sg_ott': '02567',
        'sg_app': '02568',
        'sg_arg': '02569',
        'sg_putt': '02564',
        'driving_distance': '101',
        'driving_accuracy': '102',
        'gir': '103',
        'scrambling': '130',
        'putting_average': '104',
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "pgatour"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })

    def get_current_season(self) -> int:
        """Get the current PGA Tour season year."""
        now = datetime.now()
        # PGA season runs roughly Oct-Aug, so if we're in Oct-Dec, it's next year's season
        if now.month >= 10:
            return now.year + 1
        return now.year

    def fetch_stat_leaderboard(self, stat_id: str, season: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch a specific statistic leaderboard from PGA Tour.

        Args:
            stat_id: The PGA Tour stat ID
            season: Season year (defaults to current)

        Returns:
            DataFrame with player stats
        """
        if season is None:
            season = self.get_current_season()

        # Try the stats page API
        url = f"https://www.pgatour.com/stats/stat.{stat_id}.y{season}.html"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML table
            tables = pd.read_html(response.text)
            if tables:
                df = tables[0]
                return df
        except Exception as e:
            print(f"    Warning: Could not fetch stat {stat_id}: {e}")

        return pd.DataFrame()

    def fetch_stats_json(self, stat_id: str, season: Optional[int] = None) -> List[Dict]:
        """
        Try to fetch stats from PGA Tour's JSON API.

        Args:
            stat_id: The PGA Tour stat ID
            season: Season year

        Returns:
            List of player stat dictionaries
        """
        if season is None:
            season = self.get_current_season()

        # PGA Tour's internal API endpoint
        url = f"https://statdata.pgatour.com/r/{stat_id}/{season}.json"

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'years' in data and data['years']:
                    return data['years'][0].get('details', [])
        except Exception:
            pass

        return []

    def scrape_all_stats(self, season: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape all relevant statistics and merge into single DataFrame.

        Args:
            season: Season year (defaults to current)

        Returns:
            DataFrame with all player stats
        """
        if season is None:
            season = self.get_current_season()

        print(f"\n  Scraping PGA Tour stats for {season} season...")

        all_stats = {}

        for stat_name, stat_id in self.STAT_IDS.items():
            print(f"    Fetching {stat_name}...", end=" ")

            # Try JSON API first
            data = self.fetch_stats_json(stat_id, season)

            if data:
                for player in data:
                    player_name = player.get('plrName', player.get('playerName', ''))
                    if not player_name:
                        continue

                    if player_name not in all_stats:
                        all_stats[player_name] = {'player_id': player_name}

                    # Get the stat value
                    value = player.get('statValue', player.get('value'))
                    if value is not None:
                        try:
                            all_stats[player_name][stat_name] = float(str(value).replace(',', ''))
                        except (ValueError, TypeError):
                            pass

                print(f"OK ({len(data)} players)")
            else:
                print("No data")

            # Rate limiting
            time.sleep(0.5)

        if not all_stats:
            print("  Warning: No stats retrieved from API")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(list(all_stats.values()))

        print(f"\n  Retrieved stats for {len(df)} players")
        return df

    def scrape_tournament_field(self, tournament_id: str) -> List[str]:
        """
        Scrape the field for an upcoming tournament.

        Args:
            tournament_id: PGA Tour tournament ID

        Returns:
            List of player names in the field
        """
        url = f"https://www.pgatour.com/tournaments/{tournament_id}/field"

        try:
            response = self.session.get(url, timeout=30)
            # Would need to parse the field from the page
            # This is a placeholder - actual implementation depends on page structure
        except Exception as e:
            print(f"Could not fetch tournament field: {e}")

        return []

    def save_scraped_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save scraped data to CSV.

        Args:
            df: DataFrame to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"pga_stats_{timestamp}.csv"

        filepath = self.raw_dir / filename
        df.to_csv(filepath, index=False)
        print(f"  Saved to {filepath}")

        return filepath

    def load_latest_scraped_data(self) -> pd.DataFrame:
        """Load the most recently scraped data file."""
        files = list(self.raw_dir.glob("pga_stats_*.csv"))

        if not files:
            return pd.DataFrame()

        # Get most recent file
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return pd.read_csv(latest)

    def update_player_stats(self) -> pd.DataFrame:
        """
        Main method to update player statistics.
        Scrapes current data and saves it.

        Returns:
            DataFrame with current player stats
        """
        print("\n" + "=" * 60)
        print("PGA TOUR DATA UPDATE")
        print("=" * 60)

        # Scrape current season stats
        df = self.scrape_all_stats()

        if df.empty:
            print("\n  Falling back to alternative scraping method...")
            df = self._scrape_alternative()

        if not df.empty:
            # Map to expected column format
            df = self._standardize_columns(df)

            # Save the data
            self.save_scraped_data(df)

            # Also save as the "current" stats file for the pipeline
            current_file = self.raw_dir / "current_player_stats.csv"
            df.to_csv(current_file, index=False)
            print(f"  Updated current stats: {current_file}")

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match our pipeline expectations."""
        column_map = {
            'sg_total': 'sg_total',
            'sg_ott': 'off_the_tee',
            'sg_app': 'approach_play',
            'sg_arg': 'short_game',
            'sg_putt': 'sg_putt',
            'driving_distance': 'driving_distance',
            'driving_accuracy': 'driving_accuracy',
            'gir': 'greens_in_regulation',
            'scrambling': 'scrambling',
            'putting_average': 'putting_average',
        }

        df = df.rename(columns=column_map)
        return df

    def _scrape_alternative(self) -> pd.DataFrame:
        """
        Alternative scraping method using ESPN or other sources.
        Fallback if PGA Tour API is unavailable.

        Fetches data from multiple ESPN endpoints and merges them.
        """
        print("  Trying ESPN Golf stats...")

        all_data = {}

        # ESPN stat endpoints to try
        stat_endpoints = [
            ('https://www.espn.com/golf/stats/player/_/stat/strokes-gained-total', 'SG:TOT', 'sg_total'),
            ('https://www.espn.com/golf/stats/player/_/stat/strokes-gained-tee-to-green', 'SG:T2G', 'sg_t2g'),
            ('https://www.espn.com/golf/stats/player/_/stat/strokes-gained-putting', 'SG:P', 'sg_putt'),
            ('https://www.espn.com/golf/stats/player', None, None),  # General stats page
        ]

        for url, stat_col, target_col in stat_endpoints:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    continue

                tables = pd.read_html(response.text)
                if len(tables) >= 2:
                    # ESPN splits data into two tables: names and stats
                    names_df = tables[0]
                    stats_df = tables[1]
                    df = pd.concat([names_df, stats_df], axis=1)

                    # Get player name column
                    name_col = 'Name' if 'Name' in df.columns else df.columns[1]

                    # Process each player
                    for _, row in df.iterrows():
                        player_name = row[name_col]
                        if pd.isna(player_name) or not isinstance(player_name, str):
                            continue

                        if player_name not in all_data:
                            all_data[player_name] = {'player_id': player_name}

                        # Extract specific SG stat if this is a SG page
                        if stat_col and target_col:
                            # Find the SG column (usually named with the stat)
                            for col in df.columns:
                                if 'AVG' in str(col).upper() or stat_col in str(col).upper():
                                    try:
                                        val = float(str(row[col]).replace('+', '').replace(',', ''))
                                        all_data[player_name][target_col] = val
                                    except (ValueError, TypeError):
                                        pass
                                    break

                        # Extract general stats from any page
                        col_map = {
                            'DDIS': 'driving_distance',
                            'DACC': 'driving_accuracy',
                            'GIR': 'greens_in_regulation',
                            'SAND': 'scrambling',
                            'PUTTS': 'putting_average',
                            'SCORE': 'scoring_average',
                            'BIRDS': 'birdies_per_round',
                            'EVNTS': 'events',
                            'RNDS': 'rounds',
                            'CUTS': 'cuts_made',
                            'TOP10': 'top_10s',
                            'WINS': 'wins',
                        }

                        for espn_col, our_col in col_map.items():
                            if espn_col in df.columns and our_col not in all_data[player_name]:
                                try:
                                    val = row[espn_col]
                                    if pd.notna(val):
                                        val_str = str(val).replace('%', '').replace(',', '')
                                        all_data[player_name][our_col] = float(val_str)
                                except (ValueError, TypeError):
                                    pass

                    if stat_col:
                        print(f"    Fetched {stat_col} data...")

            except Exception as e:
                print(f"    Failed to fetch {url}: {e}")
                continue

            time.sleep(0.5)  # Rate limiting

        if not all_data:
            print("  No ESPN data retrieved")
            return pd.DataFrame()

        df = pd.DataFrame(list(all_data.values()))

        # Estimate missing SG components if we have some data
        if 'sg_total' in df.columns and 'sg_t2g' in df.columns:
            # SG:Total = SG:T2G + SG:Putting
            if 'sg_putt' not in df.columns:
                df['sg_putt'] = df['sg_total'] - df['sg_t2g']

        # Estimate SG components from traditional stats if SG not available
        if 'sg_total' not in df.columns:
            # Rough estimate: good stats = positive SG
            if 'scoring_average' in df.columns:
                tour_avg = df['scoring_average'].median()
                df['sg_total'] = (tour_avg - df['scoring_average']) / 1.5
            else:
                df['sg_total'] = 0

        print(f"  Retrieved {len(df)} players from ESPN")
        print(f"  Columns: {list(df.columns)}")

        return df


    def scrape_player_tournament_history(self, player_name: str) -> pd.DataFrame:
        """
        Scrape a player's recent tournament history from ESPN.

        Args:
            player_name: Player name to search for

        Returns:
            DataFrame with tournament results including date, course, position, etc.
        """
        print(f"\n  Fetching tournament history for {player_name}...")

        # First, search for the player to get their ESPN ID
        search_url = f"https://www.espn.com/golf/player/results/_/name/{player_name.lower().replace(' ', '-')}"

        try:
            response = self.session.get(search_url, timeout=30)
            if response.status_code != 200:
                # Try alternative search
                search_url = f"https://site.web.api.espn.com/apis/common/v3/search?query={player_name}&limit=5&type=player&sport=golf"
                response = self.session.get(search_url, timeout=30)

            # Try to parse tournament results from player page
            tables = pd.read_html(response.text)
            if tables:
                # Find the results table
                for table in tables:
                    if 'DATE' in table.columns or 'TOURNAMENT' in table.columns or len(table.columns) > 5:
                        return self._parse_tournament_results(table, player_name)

        except Exception as e:
            print(f"    Could not fetch player history: {e}")

        return pd.DataFrame()

    def _parse_tournament_results(self, df: pd.DataFrame, player_name: str) -> pd.DataFrame:
        """Parse tournament results table from ESPN."""
        results = []

        # Map common column names
        col_map = {
            'DATE': 'date',
            'TOURNAMENT': 'tournament_name',
            'POS': 'position',
            'SCORE': 'score',
            'R1': 'round_1',
            'R2': 'round_2',
            'R3': 'round_3',
            'R4': 'round_4',
            'EARNINGS': 'earnings',
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for _, row in df.iterrows():
            result = {'player_id': player_name}

            for col in ['date', 'tournament_name', 'position', 'score']:
                if col in row.index and pd.notna(row[col]):
                    result[col] = row[col]

            if 'tournament_name' in result:
                results.append(result)

        return pd.DataFrame(results)

    def scrape_masters_results(self, years: List[int] = None) -> pd.DataFrame:
        """
        Scrape Masters Tournament results for specified years.

        Args:
            years: List of years to scrape (default: 2023-2025)

        Returns:
            DataFrame with Masters results
        """
        if years is None:
            years = [2023, 2024, 2025]

        print(f"\n  Scraping Masters results for {years}...")

        all_results = []

        for year in years:
            try:
                # ESPN Masters leaderboard URL pattern
                url = f"https://www.espn.com/golf/leaderboard/_/tournamentId/401580/{year}"

                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    # Try alternative URL patterns
                    alt_urls = [
                        f"https://www.espn.com/golf/leaderboard?tournamentId=401580&season={year}",
                        f"https://www.espn.com/golf/story/_/id/masters-{year}-leaderboard",
                    ]
                    for alt_url in alt_urls:
                        response = self.session.get(alt_url, timeout=30)
                        if response.status_code == 200:
                            break

                tables = pd.read_html(response.text)

                for table in tables:
                    if len(table) > 10:  # Likely the leaderboard
                        table['year'] = year
                        table['tournament_name'] = 'Masters Tournament'
                        table['course'] = 'Augusta National Golf Club - Augusta, GA'
                        all_results.append(table)
                        print(f"    {year}: Found {len(table)} players")
                        break

            except Exception as e:
                print(f"    {year}: Failed - {e}")

            time.sleep(1)  # Rate limiting

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            return self._standardize_tournament_results(combined)

        return pd.DataFrame()

    def _standardize_tournament_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize tournament results to match our data format."""
        # Common column mappings
        col_map = {
            'POS': 'position',
            'PLAYER': 'player_id',
            'Player': 'player_id',
            'Name': 'player_id',
            'TO PAR': 'score_to_par',
            'SCORE': 'total_score',
            'TOT': 'total_score',
            'THRU': 'thru',
            'R1': 'round_1',
            'R2': 'round_2',
            'R3': 'round_3',
            'R4': 'round_4',
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Parse position (handle "T5", "CUT", etc.)
        if 'position' in df.columns:
            df['position_numeric'] = pd.to_numeric(
                df['position'].astype(str).str.replace('T', '').str.replace('CUT', '999').str.replace('WD', '999'),
                errors='coerce'
            )
            df['made_cut'] = (df['position_numeric'] < 999).astype(int)
            df['top_10'] = (df['position_numeric'] <= 10).astype(int)
            df['win'] = (df['position_numeric'] == 1).astype(int)

        return df

    def load_recent_tournament_data(self) -> pd.DataFrame:
        """Load any cached recent tournament data."""
        recent_file = self.raw_dir / "recent_tournament_results.csv"
        if recent_file.exists():
            return pd.read_csv(recent_file)
        return pd.DataFrame()

    def save_recent_tournament_data(self, df: pd.DataFrame):
        """Save recent tournament data to cache."""
        recent_file = self.raw_dir / "recent_tournament_results.csv"
        df.to_csv(recent_file, index=False)
        print(f"  Saved recent tournament data to {recent_file}")


def update_data():
    """Convenience function to update PGA Tour data."""
    scraper = PGATourScraper()
    return scraper.update_player_stats()


if __name__ == "__main__":
    # Run the scraper directly
    df = update_data()

    if not df.empty:
        print("\n" + "=" * 60)
        print("DATA UPDATE COMPLETE")
        print("=" * 60)
        print(f"\nPlayers: {len(df)}")
        print(f"Stats available: {list(df.columns)}")
        print("\nSample data:")
        print(df.head(10).to_string())
    else:
        print("\nNo data retrieved. Check your internet connection.")
