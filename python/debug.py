import cloudscraper
from bs4 import BeautifulSoup

def debug_ipowatch():
    print("🔍 DEBUG MODE: Connecting to IPOWatch...")
    url = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"
    scraper = cloudscraper.create_scraper()
    
    try:
        r = scraper.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Find the main GMP table
        tables = soup.find_all('table')
        target_table = None
        for t in tables:
            if "GMP" in t.text and "IPO" in t.text:
                target_table = t
                break
        
        if not target_table:
            print("❌ No GMP table found.")
            return

        # 1. PRINT HEADERS
        print("\n--- TABLE HEADERS ---")
        headers = [th.text.strip() for th in target_table.find_all('th')]
        for i, h in enumerate(headers):
            print(f"Column [{i}]: {h}")

        # 2. PRINT FIRST 3 ROWS (RAW)
        print("\n--- FIRST 3 ROWS (DATA) ---")
        rows = target_table.find_all('tr')[1:4] # Skip header, get next 3
        for i, row in enumerate(rows):
            cols = [td.text.strip() for td in row.find_all('td')]
            print(f"Row {i+1}: {cols}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_ipowatch()
