from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest

class AccountInfo:
    def __init__(self, cash: float, buying_power: float, portfolio_value: float):
        self.Cash = cash
        self.BuyingPower = buying_power
        self.PortfolioValue = portfolio_value

class Equity:
    def __init__(self, symbol: str, name: str):
        self.Symbol = symbol
        self.Name = name

class Position:
    def __init__(self, symbol: str, qty: float, current_price: float):
        self.Symbol = symbol
        self.Quantity = qty
        self.CurrentPrice = current_price
        self.MarketValue = qty * current_price

class Transaction:
    def __init__(self, symbol: str, qty: float, side: str, fill_price: float, status: str, created_at: str):
        self.Symbol = symbol
        self.Quantity = qty
        self.FillPrice = fill_price
        self.Side = side
        self.Status = status
        self.CreatedAt = created_at

class Account:
    def __init__(self, key: str, secret: str):
        self.__client = TradingClient(key, secret, paper=True)

    def get_account_info(self) -> AccountInfo:
        account = self.__client.get_account()
        return AccountInfo(
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value)
        )
    
    def equity_tradable(self, symbol: str) -> bool:
        asset = self.__client.get_asset(symbol)
        return asset.tradable
    
    def get_all_positions(self) -> list[Position]:
        positions = [Position(pos.symbol, float(pos.qty), float(pos.current_price)) for pos in self.__client.get_all_positions()]
        return positions

    def trade_market_order(self, symbol: str, qty: float, side: OrderSide) -> Transaction | str:

        try:
            order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY)
            order = self.__client.submit_order(order_request)
            return Transaction(
                symbol=order.symbol,
                qty=float(order.filled_qty),
                side=order.side.name,
                fill_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                status=order.status.name,
                created_at=order.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        except Exception as e:
            return str(e)


    def get_transaction_history(self, limit: int = 50, status: QueryOrderStatus = QueryOrderStatus.ALL) -> list[Transaction]:
        request_params = GetOrdersRequest(status=status, limit=limit, nested=True)
        orders = self.__client.get_orders(request_params)
        orders = [Transaction(
            symbol=order.symbol,
            qty=float(order.filled_qty),
            side=order.side.name,
            fill_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
            status=order.status.name,
            created_at=order.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ) for order in orders]

        return orders
        
        
    
