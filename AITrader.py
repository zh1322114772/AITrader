import os
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide
from dotenv import load_dotenv
from AccountWrapper import Account
from MarketData import MarketData
from openai import OpenAI
from datetime import datetime
import time
import dspy
import json
import colorama
from colorama import Fore, Style

#environment setup
colorama.init()
load_dotenv()
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
APLACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
os.environ['OPENAI_API_KEY'] = OPENAI_APIKEY 

#clients setup
lm = dspy.LM("openai/gpt-5-nano", api_key = OPENAI_APIKEY)
dspy.configure(lm=lm)
openAIClient = OpenAI()

account = Account(APLACA_KEY, ALPACA_SECRET)
md = MarketData(APLACA_KEY, ALPACA_SECRET)


#objective setup
objective = '''
Follow the rules strictly and respond compactly in the required JSON format.

You are an elite professional equity investment agent. Your mission is to maximize long-term, risk-adjusted returns by actively deploying capital into individual stocks based on real, verifiable information, while keeping the portfolio resilient during market stress.

CORE PRINCIPLE: NEWS-FIRST, STOCK-FIRST, ALWAYS INVESTED
- The default action is to INVEST, not wait.
- The default instrument is INDIVIDUAL STOCKS, not ETFs.
- Cash is not a position; it is only a minimal operational buffer.

PRIMARY DECISION RULE: NEWS → STOCK SELECTION
- You MUST check the latest relevant news before every decision.
- Actively search for company-specific catalysts (earnings, guidance, M&A, regulatory changes, product launches, sector rotation).
- When credible news or data identifies specific companies as beneficiaries or losers, you MUST express this view through individual stock positions.

MANDATORY DECISION LADDER
1) If there is actionable, company- or sector-specific news:
   - Select individual stocks directly affected by the news.
   - Size positions according to conviction and risk.
   - Individual stock selection is REQUIRED in this case; ETFs are NOT allowed.

2) If news is mixed, weak, or points to a theme but not a clear single winner:
   - Construct a small basket of individual stocks that best express the theme.
   - Prefer high-quality, liquid, profitable leaders within the sector.
   - ETFs are allowed ONLY if a stock basket cannot reasonably express the view.

3) ETFs are a LAST RESORT fallback:
   - Use ETFs ONLY when:
     - No specific company can be identified with reasonable confidence, OR
     - Available information is too diffuse or macro-level only.
   - When used, select broad, liquid ETFs aligned with the identified theme or risk profile.
   - Explicitly state why individual stock selection was not feasible.

CASH IS ALLOWED ONLY IF:
- The market is closed (then prepare orders for the next open), OR
- There is a clearly identified extreme systemic event (halts, liquidity freeze).
- In all cases, provide a concrete redeployment plan.

PORTFOLIO CONSTRUCTION
- Maintain diversification across sectors and styles, but express views primarily via individual equities.
- Avoid unnecessary over-concentration in a single name.
- Favor liquid, durable, fundamentally strong companies.

EXECUTION REQUIREMENTS
- Always confirm current Eastern Time and whether the market is open.
- Never place trades when the market is closed.
- Always consult the latest relevant news immediately before execution.
- If the market is closed, sleep until near the next open.

OPERATIONAL CADENCE
- Act only when information meaningfully changes the investment outlook.
- Consolidate decisions rather than reacting continuously.
- Prefer fewer, higher-quality actions over frequent small adjustments.
- Think carefully and try to limit actions to no more than once per hour unless urgent news breaks.

FAILURE CONDITIONS
- Defaulting to ETFs without first attempting stock selection is a failure.
- Holding cash due to uncertainty is a failure.
- Acting without checking news is a failure.

RETURN FORMAT
- Always return JSON with:
  - "summary": detailed reasoning, including referenced news themes
  - "next_action_seconds": time to wait before the next action
'''

def tool_account_info() -> dict:
    """Get current account info. includes cash, buying power, portfolio value."""
    info = account.get_account_info()
    return {"cash": info.Cash, "buying_power": info.BuyingPower, "portfolio_value": info.PortfolioValue}

def tool_positions() -> list[dict]:
    """List current account's positions."""
    pos = account.get_all_positions()
    return [{"symbol": p.Symbol, "qty": p.Quantity, "price": p.CurrentPrice, "mv": p.MarketValue} for p in pos]

def tool_latest_quote(symbol: str) -> dict:
    """Get latest bid/ask quote for a symbol."""
    return md.latest_quote(symbol)

def tool_historical_quotes_last_10m(symbol: str) -> list[dict]:
    """Get historical bid/ask quotes for the last 10 minutes for a symbol."""
    return md.historical_quotes_last_10m(symbol)

def tool_place_market_order(symbol: str, qty: float, side: str) -> dict:
    """Place a market order. side must be 'buy' or 'sell'. qty is the quantity of shares(can be fractional)."""
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    res = account.trade_market_order(symbol, qty, side_enum)
    if isinstance(res, str):
        return {"ok": False, "error": res}
    
    return {
        "ok": True,
        "symbol": res.Symbol,
        "qty": res.Quantity,
        "side": res.Side,
        "fill_price": res.FillPrice,
        "status": res.Status,
        "created_at": res.CreatedAt,
    }

def tool_recent_orders(limit: int = 20) -> list[dict]:
    """Get recent orders (fills + statuses) / transactions. limit is the max number of orders to return. """
    orders = account.get_transaction_history(limit=limit)
    return [{
        "symbol": o.Symbol, "qty": o.Quantity, "side": o.Side,
        "fill_price": o.FillPrice, "status": o.Status, "created_at": o.CreatedAt
    } for o in orders]

def tool_is_tradable(symbol: str) -> bool:
    """Check if a symbol is tradable."""
    return account.equity_tradable(symbol)

def tool_search_internet(prompt: str) -> str:
    """this is a tool to search the internet for current news and information. use it to stay updated with market news. the underlying model has web search capabilities. form a good prompt to get relevant information."""

    response = openAIClient.responses.create(
        model="gpt-5-nano",
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    return response.output_text


def tool_get_current_time() -> str:
    """Get the current time in EST. use to determine time sensitive actions. like market open/close."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_app_settings():
    """Read application settings from a JSON file."""
    settings_path = "app_settings.json"
    if not os.path.exists(settings_path):
        return {}
    
    with open(settings_path, "r") as f:
        settings = json.load(f)
    
    return settings

def write_app_settings(settings: dict):
    """Write application settings to a JSON file."""
    settings_path = "app_settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

def get_log_color(role: str, type_: str) -> str:
    """Get color code for logging based on role and type."""
    if role == "System":
        if type_ == "Pause":
            return Fore.BLUE + Style.BRIGHT
    elif role == "Trader":
        if type_.startswith("Thought"):
            return Fore.YELLOW + Style.BRIGHT
        elif type_ == "Summary":
            return Fore.GREEN + Style.BRIGHT
        elif type_ == "Account":
            return Fore.CYAN + Style.BRIGHT
    return Fore.WHITE

def logger(role:str, type:str, message: str):
    """Log messages with a role and type."""
    ttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = get_log_color(role, type)
    print(f"[{ttime}]{color}[{role}]{Style.RESET_ALL}[{type}]: {message}")

class TradeSignature(dspy.Signature):
    objective: str = dspy.InputField(desc="The goal of the trading agent.")
    state: str = dspy.InputField(desc="This provides the current state of the environment, Account's info, current holdings, and current estern time in JSON.")
    action_json: str = dspy.OutputField(desc="A JSON: {summary, next_action_seconds} where 'summary' explains the reasoning behind the decision(explain everything in details). next_action_seconds is the number of seconds to wait before the next action.")


agent = dspy.ReAct(
    signature=TradeSignature,
    tools=[tool_account_info, tool_positions, tool_latest_quote, tool_place_market_order, tool_recent_orders, tool_is_tradable, tool_search_internet, tool_get_current_time, tool_historical_quotes_last_10m],
)

app_settings = read_app_settings()
next_action_stamp = app_settings.get("next_action_stamp", int(time.time()))

logger("System", "Startup", "AI Trader is starting up.")

def run_once():
    global next_action_stamp
    global app_settings

    #avoid frequent actions
    if time.time() < next_action_stamp:
        sleep_duration = next_action_stamp - int(time.time())
        logger("System", "Pause", f"Sleeping for {sleep_duration} seconds until next action.")
        time.sleep(sleep_duration)


    #ReAct step
    state = {
        "account": tool_account_info(),
        "positions": tool_positions(),
        "recent_orders": tool_recent_orders(limit=100),
    }
    state_json = json.dumps(state)
    
    pred = agent(objective=objective, state=state_json)
    result = json.loads(pred.action_json)


    #process result
    traj = pred.trajectory 
    i = 0
    while f"tool_name_{i}" in traj:
        logger("Trader", f"Thought_{i}", traj.get(f"thought_{i}"))
        i += 1

    summary : str = "Error in parsing result."
    wait_time : int = 60

    if 'summary' in result:
        summary = result['summary']

    if 'next_action_seconds' in result:
        wait_time = result['next_action_seconds']
        wait_time = max(10, wait_time)  #minimum wait time of 30 seconds
    
    
    #update next action timestamp and write to settings
    next_action_stamp = int(time.time()) + wait_time
    app_settings["next_action_stamp"] = next_action_stamp
    write_app_settings(app_settings)

    
    logger("Trader", "Summary", summary)


    #pull account info after action
    updated_account_info = tool_account_info()
    logger("Trader", "Account", f"Cash: {updated_account_info['cash']}, Buying Power: {updated_account_info['buying_power']}, Portfolio Value: {updated_account_info['portfolio_value']}")

while True:

    try:
        run_once()
    except Exception as e:
        logger("System", "Error", f"An error occurred: {str(e)}")
        time.sleep(60)  #wait for a minute before retrying
