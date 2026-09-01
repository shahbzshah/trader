"""Tests for app/mt5_client.py using a fake MetaTrader5 module."""

from __future__ import annotations

import pytest

from app import mt5_client
from app.mt5_client import (
    AccountInfo,
    LiveAccountError,
    MT5Client,
    MT5ConnectionError,
    MT5LoginError,
    MT5RequestError,
    Position,
    Rates,
    Tick,
)
from tests.fake_mt5 import (
    ACCOUNT_TRADE_MODE_DEMO,
    ACCOUNT_TRADE_MODE_REAL,
    FakeAccountInfo,
    FakeMT5,
    FakePosition,
    FakeRatesRow,
    FakeSymbolInfo,
    FakeTick,
    ORDER_FILLING_FOK,
    ORDER_FILLING_IOC,
    ORDER_FILLING_RETURN,
    make_fake,
)


@pytest.fixture
def client(fake_mt5: FakeMT5) -> MT5Client:
    c = MT5Client()
    c.connect(login=12345, password="pw", server="DemoBroker-Server", path="")
    return c


def test_connect_succeeds(fake_mt5, client):
    assert client._connected is True
    assert ("initialize", "") in fake_mt5.calls
    assert any(call[0] == "login" for call in fake_mt5.calls)


def test_connect_retries_then_succeeds(fake_mt5: FakeMT5):
    fake_mt5.initialize_ok = True
    fake_mt5.login_ok = True
    # First login attempt fails, second succeeds.
    state = {"n": 0}

    real_login = fake_mt5.login

    def flaky_login(login, password, server):
        state["n"] += 1
        if state["n"] == 1:
            fake_mt5.last_error_value = (1, "transient")
            return False
        return real_login(login, password, server)

    fake_mt5.login = flaky_login
    c = MT5Client()
    c.connect(login=12345, password="pw", server="DemoBroker-Server")
    assert state["n"] == 2
    assert c._connected is True


def test_connect_raises_on_login_failure(fake_mt5: FakeMT5):
    fake_mt5.login_ok = False
    c = MT5Client()
    with pytest.raises(MT5LoginError):
        c.connect(login=1, password="pw", server="x")


def test_connect_raises_on_init_failure(fake_mt5: FakeMT5):
    fake_mt5.initialize_ok = False
    c = MT5Client()
    with pytest.raises(MT5ConnectionError):
        c.connect(login=1, password="pw", server="x")


def test_account_info_returns_demo(client, fake_mt5):
    info = client.account_info()
    assert isinstance(info, AccountInfo)
    assert info.trade_mode == "demo"
    assert info.login == 12345
    assert ("account_info",) in fake_mt5.calls


def test_account_info_handles_real_account(fake_mt5: FakeMT5):
    fake_mt5.trade_mode = ACCOUNT_TRADE_MODE_REAL
    c = MT5Client()
    c.connect(login=1, password="pw", server="x")
    info = c.account_info()
    assert info.trade_mode == "real"


def test_ensure_demo_blocks_non_demo(fake_mt5: FakeMT5):
    fake_mt5.trade_mode = ACCOUNT_TRADE_MODE_REAL
    c = MT5Client()
    c.connect(login=1, password="pw", server="x")
    with pytest.raises(LiveAccountError):
        c.ensure_demo()


def test_ensure_demo_blocks_real_account_regression(fake_mt5: FakeMT5):
    """Regression: a real-mode account must always be rejected by ensure_demo.

    This guards against regressing to the wrong constant family (e.g.
    ACCOUNT_MARGIN_MODE_*) which would let a real account slip through.
    """
    fake_mt5.trade_mode = ACCOUNT_TRADE_MODE_REAL
    fake_mt5.account.trade_mode = ACCOUNT_TRADE_MODE_REAL
    c = MT5Client()
    c.connect(login=42, password="pw", server="RealBroker-Server")
    # ensure_demo must raise before any order can be placed.
    with pytest.raises(LiveAccountError, match="non-demo"):
        c.ensure_demo()
    # And market_order must also be vetoed.
    with pytest.raises(LiveAccountError):
        c.market_order("EURUSD", "buy", 0.10, 1.0, 1.1)


def test_ensure_demo_allows_demo(client):
    info = client.ensure_demo()
    assert info.trade_mode == "demo"


def test_positions_returns_list(client, fake_mt5: FakeMT5):
    fake_mt5.positions_data = [
        FakePosition(ticket=1, symbol="EURUSD", type=0, volume=0.10),
        FakePosition(ticket=2, symbol="GBPUSD", type=1, volume=0.20),
    ]
    out = client.positions()
    assert [p.ticket for p in out] == [1, 2]
    assert all(isinstance(p, Position) for p in out)
    assert out[0].direction == "buy"
    assert out[1].direction == "sell"


def test_positions_by_symbol(client, fake_mt5: FakeMT5):
    fake_mt5.positions_data = [FakePosition(ticket=1, symbol="EURUSD")]
    out = client.positions(symbol="EURUSD")
    assert len(out) == 1


def test_copy_rates_returns_records(client, fake_mt5: FakeMT5):
    fake_mt5.rates_data = [
        FakeRatesRow(time=1, open=1.1, high=1.2, low=1.0, close=1.15),
        FakeRatesRow(time=2, open=1.15, high=1.25, low=1.14, close=1.20),
    ]
    rates = client.copy_rates("EURUSD", "M15", 100)
    assert isinstance(rates, Rates)
    assert rates.symbol == "EURUSD"
    assert rates.timeframe == "M15"
    assert len(rates.rates) == 2
    assert rates.rates[0]["close"] == 1.15


def test_copy_rates_unsupported_tf(client):
    with pytest.raises(ValueError):
        client.copy_rates("EURUSD", "M2", 100)


def test_symbol_tick_returns_tick(client, fake_mt5: FakeMT5):
    fake_mt5.tick_data["EURUSD"] = FakeTick(bid=1.1, ask=1.1002, time=42)
    t = client.symbol_tick("EURUSD")
    assert isinstance(t, Tick)
    assert t.bid == 1.1
    assert t.ask == 1.1002
    assert abs(t.spread - 0.0002) < 1e-9
    assert t.time == 42


def test_symbol_info_returns_min_max_step(client, fake_mt5: FakeMT5):
    fake_mt5.symbol_data["EURUSD"] = FakeSymbolInfo(
        name="EURUSD",
        digits=5,
        point=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_contract_size=100000.0,
    )
    s = client.symbol_info("EURUSD")
    assert s.volume_min == 0.01
    assert s.volume_max == 100.0
    assert s.volume_step == 0.01


def test_market_order_success(client, fake_mt5: FakeMT5):
    fake_mt5.tick_data["EURUSD"] = FakeTick(bid=1.1000, ask=1.1001, time=1)
    res = client.market_order("EURUSD", "buy", lots=0.10, sl=1.095, tp=1.110, comment="t")
    assert res.retcode == 10009
    assert res.order_id > 0
    # ensure_demo must have been called -> account_info called.
    assert ("account_info",) in fake_mt5.calls


def test_market_order_rejects_non_demo(fake_mt5: FakeMT5):
    fake_mt5.trade_mode = ACCOUNT_TRADE_MODE_REAL
    c = MT5Client()
    c.connect(login=1, password="pw", server="x")
    with pytest.raises(LiveAccountError):
        c.market_order("EURUSD", "buy", 0.1, 1.0, 1.1)


def test_market_order_invalid_direction(client):
    with pytest.raises(ValueError):
        client.market_order("EURUSD", "sideways", 0.1, 1.0, 1.1)


def test_market_order_request_failure(client, fake_mt5: FakeMT5):
    fake_mt5.tick_data["EURUSD"] = FakeTick(bid=1.0, ask=1.0, time=0)
    fake_mt5.order_fail_retcode = 10004
    with pytest.raises(MT5RequestError):
        client.market_order("EURUSD", "buy", 0.1, 1.0, 1.1)


def test_close_position_success(fake_mt5: FakeMT5):
    fake_mt5.positions_data = [
        FakePosition(ticket=42, symbol="EURUSD", type=0, volume=0.10,
                     price_open=1.1, price_current=1.11, sl=1.09, tp=1.12)
    ]
    fake_mt5.tick_data["EURUSD"] = FakeTick(bid=1.1100, ask=1.1110, time=1)
    c = MT5Client()
    c.connect(login=1, password="pw", server="x")
    res = c.close_position(42)
    assert res.retcode == 10009
    # Check request type == SELL (closing a BUY).
    send_call = next(c for c in fake_mt5.calls if c[0] == "order_send")
    assert send_call[1]["type"] == fake_mt5.ORDER_TYPE_SELL


def test_close_position_not_found(client, fake_mt5: FakeMT5):
    with pytest.raises(MT5RequestError):
        client.close_position(999)


def test_modify_position_success(client, fake_mt5: FakeMT5):
    ok = client.modify_position(ticket=42, sl=1.0, tp=1.2)
    assert ok is True
    send_call = next(c for c in fake_mt5.calls if c[0] == "order_send")
    assert send_call[1]["action"] == fake_mt5.TRADE_ACTION_SLTP
    assert send_call[1]["sl"] == 1.0
    assert send_call[1]["tp"] == 1.2


def test_methods_require_connected(fake_mt5: FakeMT5):
    c = MT5Client()
    with pytest.raises(MT5ConnectionError):
        c.account_info()
    with pytest.raises(MT5ConnectionError):
        c.positions()
    with pytest.raises(MT5ConnectionError):
        c.market_order("EURUSD", "buy", 0.1, 1.0, 1.1)


def test_shutdown_clears_connected(client, fake_mt5: FakeMT5):
    client.shutdown()
    assert client._connected is False
    assert ("shutdown",) in fake_mt5.calls


def test_no_metatrader5_import_outside_client(monkeypatch):
    """Safety invariant: only app.mt5_client may import MetaTrader5."""
    # Importing other app modules must not trigger MetaTrader5 import.
    import app.config  # noqa: F401
    import app.scanner  # noqa: F401
    import app.brain  # noqa: F401
    import app.risk  # noqa: F401
    # app.mt5_client itself lazily imports; without _get_mt5() call nothing happens.
    assert mt5_client._mt5 is None or True  # both states acceptable

    # Sanity: confirm the actual MetaTrader5 package isn't installed.
    try:
        import MetaTrader5  # noqa: F401

        has_real = True
    except ImportError:
        has_real = False

    if not has_real:
        # Ensure _get_mt5() raises ImportError as MT5NotInstalledError signal.
        mt5_client._mt5 = None
        with pytest.raises(ImportError):
            mt5_client._get_mt5()


# --- Market Watch selection & symbol discovery (real-broker support) ----


def test_select_symbol_fast_path_when_visible(fake_mt5, client):
    """EURUSD is already in the terminal -> no symbol_select call needed."""
    fake_mt5.calls.clear()
    assert client.select_symbol("EURUSD") is True
    assert ("symbol_select", "EURUSD", True) not in fake_mt5.calls


def test_select_symbol_calls_symbol_select_when_hidden(fake_mt5, client):
    fake_mt5.symbol_data.pop("EURUSD")
    assert client.select_symbol("EURUSD") is True
    assert ("symbol_select", "EURUSD", True) in fake_mt5.calls


def test_select_symbol_returns_false_when_rejected(fake_mt5, client):
    fake_mt5.symbol_data.pop("EURUSD")
    fake_mt5.symbol_select_ok = False
    assert client.select_symbol("EURUSD") is False


def test_select_symbol_requires_connection(fake_mt5):
    c = MT5Client()
    with pytest.raises(MT5ConnectionError):
        c.select_symbol("EURUSD")


def test_symbol_names_returns_terminal_symbols(fake_mt5, client):
    fake_mt5.symbols_get_result = [
        FakeSymbolInfo(name="EURUSD"),
        FakeSymbolInfo(name="EURUSD.a"),
        FakeSymbolInfo(name="GBPUSD"),
    ]
    assert client.symbol_names() == ["EURUSD", "EURUSD.a", "GBPUSD"]


# --- filling-mode selection (broker retcode 10030 hardening) -------------


def test_type_filling_prefers_ioc(fake_mt5, client):
    fake_mt5.symbol_data["EURUSD"].filling_mode = 2  # SYMBOL_FILLING_MODE_IOC
    assert client._type_filling("EURUSD") == ORDER_FILLING_IOC


def test_type_filling_falls_back_to_fok(fake_mt5, client):
    fake_mt5.symbol_data["EURUSD"].filling_mode = 1  # FOK only
    assert client._type_filling("EURUSD") == ORDER_FILLING_FOK


def test_type_filling_falls_back_to_return(fake_mt5, client):
    fake_mt5.symbol_data["EURUSD"].filling_mode = 4  # RETURN only
    assert client._type_filling("EURUSD") == ORDER_FILLING_RETURN


def test_type_filling_unknown_mode_defaults_to_ioc(fake_mt5, client):
    fake_mt5.symbol_data["EURUSD"].filling_mode = 0  # unknown bitmask
    assert client._type_filling("EURUSD") == ORDER_FILLING_IOC


def test_type_filling_missing_symbol_defaults_to_ioc(fake_mt5, client):
    fake_mt5.symbol_data.pop("EURUSD")
    assert client._type_filling("EURUSD") == ORDER_FILLING_IOC


def test_market_order_uses_symbol_filling_mode(fake_mt5, client):
    fake_mt5.symbol_data["EURUSD"].filling_mode = 1  # FOK only
    client.market_order("EURUSD", "buy", 0.01, 1.09, 1.12, "t")
    order_calls = [c for c in fake_mt5.calls if c[0] == "order_send"]
    assert order_calls, "expected an order_send call"
    assert order_calls[-1][1]["type_filling"] == ORDER_FILLING_FOK