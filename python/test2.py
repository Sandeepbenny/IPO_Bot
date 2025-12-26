import os
import json
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# ==========================================
# 1. CONFIGURATION
# ==========================================
os.environ["GROQ_API_KEY"] 

# Initialize Model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ==========================================
# 2. THE BROWSER TOOL (Decorator Syntax)
# ==========================================
import re

@tool
def fetch_ipo_gmp_tool(dummy: str = "") -> str:
    """
    Goes to IPOWatch.in, extracts the latest IPO GMP table, and returns it as JSON.
    """
    print(f"\n🕵️ TOOL CALL: Launching Browser to fetch IPO Data...")
    
    url = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            page.goto(url, timeout=60000)
            
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            target_table = None
            for table in soup.find_all('table'):
                if "GMP" in table.text and "IPO" in table.text:
                    target_table = table
                    break
            
            if not target_table: return "Error: No GMP table found."
            
            data = []
            rows = target_table.find_all('tr')[1:10] # Top 10 rows
            
            for row in rows:
                cols = [td.text.strip() for td in row.find_all('td')]
                
                # SMART PARSING: Don't trust column order. Trust content.
                company = "Unknown"
                price = "N/A"
                gmp = "N/A"
                
                for col in cols:
                    # Is it a Price (starts with ₹)?
                    if "₹" in col:
                        # If we already have a price, the second one is likely GMP
                        if price != "N/A": 
                            gmp = col
                        else:
                            price = col
                    # Is it a Percentage?
                    elif "%" in col:
                        pass # Ignore % column
                    # If it's not a price and not a date, it's the Company Name
                    elif len(col) > 3 and not re.search(r'\d', col): # Text with no numbers
                        company = col
                
                # Fallback: If regex failed, use standard positions
                if company == "Unknown" and len(cols) >= 3:
                     company = cols[0]
                     if "₹" in company: # If first col is price, shift everything
                         company = "Unknown (Parse Error)"

                if company != "Unknown" and "SME" not in company:
                    data.append({
                        "Company": company,
                        "Price": price,
                        "GMP": gmp
                    })
            
            print(f"   -> Extracted {len(data)} IPOs.")
            return json.dumps(data, indent=2)
            
        except Exception as e: return f"Error: {e}"
        finally: browser.close()

# ==========================================
# 3. THE AGENT (LangGraph)
# ==========================================

def run_agent():
    print("==========================================")
    print("  🤖 IPO AGENT (LangGraph Edition)       ")
    print("==========================================\n")
    
    # 1. Create the Graph
    tools = [fetch_ipo_gmp_tool]
    agent_executor = create_react_agent(llm, tools)
    
    # 2. Define the Query
    query = """
    Go and fetch the latest IPO GMP list. 
    Then, for the top 3 IPOs with the highest GMP, give me a verdict (BUY/AVOID).
    
    Rules:
    - BUY if GMP is strong.
    - AVOID if GMP is negative or zero.
    
    Output a nice summary.
    """
    
    # 3. Run It
    # LangGraph returns a stream of events. We just want the final answer.
    print("🧠 Thinking...")
    events = agent_executor.stream(
        {"messages": [("user", query)]},
        stream_mode="values"
    )
    
    for event in events:
        # Print the last message from the AI
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai":
                print(f"\n💬 AGENT: {last_msg.content}")

if __name__ == "__main__":
    run_agent()
