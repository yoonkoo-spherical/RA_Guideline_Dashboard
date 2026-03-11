import os
import time
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping via ScraperAPI ---")
    
    for keyword in keywords:
        target_url = f"https://www.fda.gov/regulatory-information/search-fda-guidance-documents?keys={keyword}"
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': target_url,
            'render': 'true'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"FDA ({keyword}) - Attempt {attempt + 1}/{max_retries}")
                response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
                
                if response.status_code == 200:
                    print(f" -> Success (Status 200)")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    table_rows = soup.select("table tbody tr")
