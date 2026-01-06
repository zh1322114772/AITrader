from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockQuotesRequest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

class MarketData:
    def __init__(self, key: str, secret: str):
        self.client = StockHistoricalDataClient(key, secret)

    def latest_quote(self, symbol: str) -> dict:
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        resp = self.client.get_stock_latest_quote(req)
        q = resp[symbol]
        return {"bid": float(q.bid_price), "ask": float(q.ask_price)}
    
    def historical_quotes_last_10m(self, symbol: str) -> List[Dict[str, float | str | None]]:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=10)

        FEED = "iex"
        LIMIT_PER_PAGE = 10000
        MAX_PAGES = 5  # plenty for 10 minutes, still a safety cap

        out: List[Dict[str, float | str | None]] = []
        page_token: Optional[str] = None
        pages = 0

        while True:
            req = StockQuotesRequest(
                symbol_or_symbols=[symbol],
                start=start_dt,
                end=end_dt,
                limit=LIMIT_PER_PAGE,
                sort="asc",
                page_token=page_token,
                feed=FEED,
            )
            quote_set = self.client.get_stock_quotes(req)

            for q in quote_set[symbol]:
                out.append({
                    "t": q.timestamp.isoformat() if getattr(q, "timestamp", None) else None,
                    "bid": float(q.bid_price) if q.bid_price is not None else None,
                    "ask": float(q.ask_price) if q.ask_price is not None else None,
                })

            page_token = getattr(quote_set, "next_page_token", None)
            pages += 1
            if not page_token or pages >= MAX_PAGES:
                break

        return out