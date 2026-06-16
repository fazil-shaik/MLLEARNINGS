import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import pickle

class CoinGeckoMiningDataFetcher:
    def __init__(self, api_key=None, use_pro_api=False):
        if use_pro_api and api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
            self.headers = {"x-cg-pro-api-key": api_key}
        else:
            self.base_url = "https://api.coingecko.com/api/v3"
            self.headers = {}
        
        self.api_key = api_key
        self.use_pro_api = use_pro_api
        self.rate_limit_delay = 3  # seconds between requests for free tier
        
        # Mining hardware specs
        self.hardware = {
            'Antminer S19 Pro': {'hashrate': 110, 'power': 3250, 'cost_usd': 2500},
            'Antminer S19 XP': {'hashrate': 140, 'power': 3010, 'cost_usd': 3800},
            'Whatsminer M50': {'hashrate': 126, 'power': 3276, 'cost_usd': 2800},
            'Antminer S21 Pro': {'hashrate': 234, 'power': 3540, 'cost_usd': 5000},
        }
        
        self.coins = ['bitcoin', 'ethereum', 'litecoin']
    
    def _make_request(self, endpoint, params=None):
        """Make API request with rate limiting"""
        url = f"{self.base_url}{endpoint}"
        time.sleep(self.rate_limit_delay)  # Respect rate limits
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 429:  # Rate limited
                print(" Rate limited, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, params=params, headers=self.headers)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None
    
    def get_coin_list(self):
        """Get all supported coins from CoinGecko"""
        endpoint = "/coins/list"
        data = self._make_request(endpoint)
        if data:
            # Convert to DataFrame for easy lookup
            df = pd.DataFrame(data)
            print(f"Retrieved {len(df)} coins from CoinGecko")
            return df
        return None
    
    def get_market_data(self, vs_currency='usd', limit=100):
        """
        Get current market data for top coins.
        Endpoint: /coins/markets
        Returns: price, market cap, volume, etc. for multiple coins
        """
        endpoint = "/coins/markets"
        params = {
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': 'false'
        }
        
        data = self._make_request(endpoint, params)
        if data:
            df = pd.DataFrame(data)
            print(f" Retrieved market data for {len(df)} coins")
            return df
        return None
    
    def get_historical_prices(self, coin_id='bitcoin', days=180):
        """
        Get historical price data.
        Endpoint: /coins/{id}/market_chart
        Returns: prices, market caps, volumes
        """
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'  # Use 'daily' for consistent daily data
        }
        
        data = self._make_request(endpoint, params)
        if data:
            # Parse response
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices['coin'] = coin_id
            
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            
            # Merge all
            df = prices
            df['market_cap'] = market_caps['market_cap']
            df['volume'] = volumes['volume']
            
            print(f" Retrieved {len(df)} historical records for {coin_id}")
            return df
        return None
    
    def get_historical_range(self, coin_id='bitcoin', from_date=None, to_date=None):
        """
        Get historical data for a specific date range.
        Endpoint: /coins/{id}/market_chart/range
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        endpoint = f"/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': from_date,
            'to': to_date
        }
        
        data = self._make_request(endpoint, params)
        if data:
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices['coin'] = coin_id
            return prices
        return None
    
    def build_mining_dataset(self, n_days=365):
        """
        Build complete mining dataset using real CoinGecko data.
        This is the main method for your project.
        """
        print("Building mining dataset from CoinGecko...")
        
        all_data = []
        
        # Get historical data for each coin
        for coin in self.coins:
            print(f"\nFetching data for {coin}...")
            df = self.get_historical_prices(coin, n_days)
            if df is not None:
                all_data.append(df)
            else:
                print(f"Failed to fetch {coin}, using synthetic fallback")
                # Fallback: generate synthetic for this coin
                fallback = self._generate_fallback_data(coin, n_days)
                all_data.append(fallback)
        
        if not all_data:
            print("No data fetched, generating full synthetic dataset")
            return self._generate_full_synthetic(n_days)
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Add mining-specific calculations
        combined = self._add_mining_metrics(combined)
        
        return combined
    
    def _generate_fallback_data(self, coin, n_days):
        """Generate synthetic data for a coin if API fails"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Different base prices for different coins
        base_prices = {
            'bitcoin': 30000,
            'ethereum': 2000,
            'litecoin': 100
        }
        base = base_prices.get(coin, 1000)
        
        # Generate price with realistic volatility
        price = base * (1 + np.linspace(0, 0.3, n_days) + np.random.normal(0, 0.1, n_days))
        price = np.maximum(price, base * 0.5)
        
        df = pd.DataFrame({
            'date': dates,
            'price': price,
            'coin': coin,
            'market_cap': price * np.random.uniform(1e6, 1e9, n_days),
            'volume': price * np.random.uniform(1e4, 1e6, n_days)
        })
        print(f"Generated fallback data for {coin}")
        return df
    
    def _generate_full_synthetic(self, n_days):
        """Generate full synthetic dataset if all API calls fail"""
        print("Generating full synthetic dataset...")
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        data = []
        for coin in self.coins:
            base = {'bitcoin': 30000, 'ethereum': 2000, 'litecoin': 100}[coin]
            price = base * (1 + np.linspace(0, 0.3, n_days) + np.random.normal(0, 0.1, n_days))
            price = np.maximum(price, base * 0.5)
            
            df = pd.DataFrame({
                'date': dates,
                'price': price,
                'coin': coin,
                'market_cap': price * np.random.uniform(1e6, 1e9, n_days),
                'volume': price * np.random.uniform(1e4, 1e6, n_days)
            })
            data.append(df)
        
        return pd.concat(data, ignore_index=True)
    
    def _add_mining_metrics(self, df):
        """
        Add mining profitability calculations to the dataset.
        This is where we add the regression targets.
        """
        np.random.seed(42)
        n = len(df)
        
        # Hardware assignment (random for demo)
        hardware_names = list(self.hardware.keys())
        hardware_idx = np.random.randint(0, len(hardware_names), n)
        hardware = [hardware_names[i] for i in hardware_idx]
        
        # Convert to numpy arrays for calculations
        hardware_array = np.array(hardware)
        hashrate_hw = np.array([self.hardware[h]['hashrate'] for h in hardware_array])
        power_consumption = np.array([self.hardware[h]['power'] for h in hardware_array])
        hardware_cost = np.array([self.hardware[h]['cost_usd'] for h in hardware_array])
        
        # Simulate network difficulty (correlated with price)
        price_values = df['price'].values
        difficulty = 50e12 * (1 + 0.5 * (price_values / 30000 - 1)) + np.random.normal(0, 5e12, n)
        difficulty = np.maximum(difficulty, 30e12)
        
        # Mining calculations
        seconds_per_day = 86400
        block_reward = 6.25  # BTC per block
        
        # BTC mined per day based on hashrate and difficulty
        btc_per_day = (hashrate_hw * 1e12 * seconds_per_day * block_reward) / (difficulty * 2**32)
        
        # Apply pool fee (1-4%)
        pool_fee = np.random.uniform(0.01, 0.04, n)
        btc_per_day = btc_per_day * (1 - pool_fee)
        
        # USD revenue
        daily_revenue = btc_per_day * price_values
        
        # Electricity costs
        electricity_cost = np.random.uniform(0.03, 0.12, n)
        daily_power_cost = (power_consumption / 1000) * 24 * electricity_cost
        
        # Hardware depreciation (daily)
        hardware_age = np.random.uniform(0, 24, n)
        hardware_depreciation = hardware_cost / (24 * 30)  # Spread over 2 years
        
        daily_total_cost = daily_power_cost + hardware_depreciation
        
        # Monthly profit
        monthly_profit = (daily_revenue - daily_total_cost) * 30
        
        # Break-even days
        break_even = hardware_cost / np.maximum(daily_revenue - daily_total_cost, 0.01)
        
        # Add to DataFrame
        df['network_difficulty'] = difficulty
        df['hashrate_ths'] = hashrate_hw
        df['hardware'] = hardware
        df['power_consumption_w'] = power_consumption
        df['electricity_cost_per_kwh'] = electricity_cost
        df['pool_fee'] = pool_fee
        df['hardware_age_months'] = hardware_age
        df['daily_revenue_usd'] = daily_revenue
        df['daily_power_cost_usd'] = daily_power_cost
        df['monthly_profit_usd'] = monthly_profit
        df['break_even_days'] = break_even
        
        return df
    
    def get_live_price(self, coin_id='bitcoin'):
        """
        Get current live price for a coin.
        Endpoint: /simple/price
        """
        endpoint = "/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        data = self._make_request(endpoint, params)
        if data and coin_id in data:
            return data[coin_id]['usd']
        return None

def save_dataset(df, filename='mining_data_real.csv'):
    df.to_csv(filename, index=False)
    print(f"💾 Dataset saved to {filename}")
    return df

def load_dataset(filename='mining_data_real.csv'):
    return pd.read_csv(filename)

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("COINGECKO MINING DATA FETCHER")
    print("="*60)
    
    # Initialize with free API (no key needed)
    fetcher = CoinGeckoMiningDataFetcher()
    
    # Test API connection
    print("\n📡 Testing API connection...")
    try:
        # Get Bitcoin price
        price = fetcher.get_live_price('bitcoin')
        if price:
            print(f"BTC/USD: ${price:,.2f}")
        else:
            print(" Could not fetch live price")
    except Exception as e:
        print(f" API test failed: {e}")
    
    # Build dataset
    print("\nBuilding historical dataset...")
    df = fetcher.build_mining_dataset(n_days=365)
    print(f"\nDataset ready: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save
    save_dataset(df)