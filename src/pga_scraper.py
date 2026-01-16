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
        """
        print("  Trying ESPN Golf stats...")

        try:
            # ESPN golf stats page with actual performance stats
            url = "https://www.espn.com/golf/stats/player/_/stat/strokes-gained-total"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                tables = pd.read_html(response.text)
                if len(tables) >= 2:
                    # ESPN splits data into two tables: names and stats
                    names_df = tables[0]  # Contains RK, Name
                    stats_df = tables[1]  # Contains the stats columns

                    # Merge them side by side
                    df = pd.concat([names_df, stats_df], axis=1)

                    # Map ESPN columns to our expected format
                    espn_column_map = {
                        'Name': 'player_id',
                        'DDIS': 'driving_distance',
                        'DACC': 'driving_accuracy',
                        'GIR': 'greens_in_regulation',
                        'SAND': 'scrambling',
                        'PUTTS': 'putting_average',
                        'SCORE': 'scoring_average',
                    }

                    df = df.rename(columns=espn_column_map)

                    # Convert percentage columns (remove % if present)
                    for col in ['driving_accuracy', 'greens_in_regulation', 'scrambling']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')

                    # Convert numeric columns
                    for col in ['driving_distance', 'putting_average', 'scoring_average']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    print(f"  Retrieved {len(df)} players from ESPN")
                    print(f"  Columns: {list(df.columns)}")
                    return df
                elif tables:
                    df = tables[0]
                    print(f"  Retrieved {len(df)} players from ESPN (basic)")
                    return df
        except Exception as e:
            print(f"  ESPN scraping failed: {e}")

        return pd.DataFrame()


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
