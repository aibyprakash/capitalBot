"""Capital.com intraday support/resistance bot.

This module fetches intraday price history from Capital.com and identifies
support/resistance zones, along with breakout and reversal signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import os
from typing import Iterable, List, Optional, Sequence

import requests


DEFAULT_BASE_URL = "https://api-capital.backend-capital.com"
DEMO_BASE_URL = "https://demo-api-capital.backend-capital.com"


def _parse_iso_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass
class SupportResistanceZone:
    level: float
    kind: str
    touches: int


@dataclass
class Signal:
    kind: str
    level: float
    timestamp: datetime
    details: str


@dataclass
class RangeContext:
    high: float
    low: float
    mid: float
    start: datetime
    end: datetime
    touches_high: int
    touches_low: int
    atr_percent: float
    ema_slope: float
    adx: Optional[float]
    quality_score: float


@dataclass
class RangeSweepSignal:
    signal: str
    reason: List[str]
    entry_price: Optional[float]
    sl_price: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    confidence_score: float
    range_high: Optional[float]
    range_low: Optional[float]
    sweep_extreme: Optional[float]
    timestamp: Optional[datetime]


class CapitalComClient:
    def __init__(self, api_key: str, identifier: str, password: str, base_url: str) -> None:
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-CAP-API-KEY": self.api_key})
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None

    def login(self) -> None:
        response = self.session.post(
            f"{self.base_url}/api/v1/session",
            json={"identifier": self.identifier, "password": self.password},
        )
        response.raise_for_status()
        self.cst = response.headers.get("CST")
        self.security_token = response.headers.get("X-SECURITY-TOKEN")
        if not self.cst or not self.security_token:
            raise RuntimeError("Missing session headers from Capital.com")
        self.session.headers.update({"CST": self.cst, "X-SECURITY-TOKEN": self.security_token})

    def get_price_history(self, epic: str, resolution: str, start: datetime, end: datetime) -> List[PriceBar]:
        params = {
            "epic": epic,
            "resolution": resolution,
            "from": start.isoformat(),
            "to": end.isoformat(),
        }
        response = self.session.get(f"{self.base_url}/api/v1/prices", params=params)
        response.raise_for_status()
        data = response.json()
        bars: List[PriceBar] = []
        for bar in data.get("prices", []):
            volume = bar.get("lastTradedVolume")
            bars.append(
                PriceBar(
                    timestamp=_parse_iso_time(bar["snapshotTimeUTC"]),
                    open=float(bar["openPrice"]["bid"]),
                    high=float(bar["highPrice"]["bid"]),
                    low=float(bar["lowPrice"]["bid"]),
                    close=float(bar["closePrice"]["bid"]),
                    volume=float(volume) if volume is not None else None,
                )
            )
        return bars

    def get_accounts(self) -> List[AccountSummary]:
        response = self.session.get(f"{self.base_url}/api/v1/accounts")
        response.raise_for_status()
        data = response.json()
        accounts: List[AccountSummary] = []
        for account in data.get("accounts", []):
            balance = account.get("balance", {})
            accounts.append(
                AccountSummary(
                    account_id=str(account.get("accountId", "")),
                    name=str(account.get("accountName", "")),
                    balance=float(balance.get("balance", 0)),
                    available=float(balance.get("available", 0)),
                )
            )
        return accounts


class SupportResistanceAnalyzer:
    def __init__(
        self,
        lookback: int = 3,
        zone_tolerance: float = 0.002,
        breakout_threshold: float = 0.001,
    ) -> None:
        self.lookback = lookback
        self.zone_tolerance = zone_tolerance
        self.breakout_threshold = breakout_threshold

    def find_swings(self, bars: Iterable[PriceBar]) -> List[tuple[str, PriceBar]]:
        bars_list = list(bars)
        swings: List[tuple[str, PriceBar]] = []
        for idx in range(self.lookback, len(bars_list) - self.lookback):
            window = bars_list[idx - self.lookback : idx + self.lookback + 1]
            current = bars_list[idx]
            max_high = max(bar.high for bar in window)
            min_low = min(bar.low for bar in window)
            if current.high >= max_high:
                swings.append(("high", current))
            if current.low <= min_low:
                swings.append(("low", current))
        return swings

    def build_zones(self, swings: Iterable[tuple[str, PriceBar]]) -> List[SupportResistanceZone]:
        zones: List[SupportResistanceZone] = []
        for kind, bar in swings:
            price = bar.high if kind == "high" else bar.low
            matched = None
            for zone in zones:
                if abs(zone.level - price) / zone.level <= self.zone_tolerance:
                    matched = zone
                    break
            if matched:
                matched.level = (matched.level * matched.touches + price) / (matched.touches + 1)
                matched.touches += 1
                if matched.kind != kind:
                    matched.kind = "mixed"
            else:
                zones.append(
                    SupportResistanceZone(
                        level=price,
                        kind="resistance" if kind == "high" else "support",
                        touches=1,
                    )
                )
        zones.sort(key=lambda zone: zone.level)
        return zones

    def detect_breakouts(self, bars: List[PriceBar], zones: Iterable[SupportResistanceZone]) -> List[Signal]:
        if len(bars) < 2:
            return []
        previous = bars[-2]
        latest = bars[-1]
        signals: List[Signal] = []
        for zone in zones:
            threshold = zone.level * self.breakout_threshold
            if zone.kind in {"resistance", "mixed"}:
                if previous.close <= zone.level and latest.close > zone.level + threshold:
                    signals.append(
                        Signal(
                            kind="breakout",
                            level=zone.level,
                            timestamp=latest.timestamp,
                            details="Bullish breakout above resistance",
                        )
                    )
            if zone.kind in {"support", "mixed"}:
                if previous.close >= zone.level and latest.close < zone.level - threshold:
                    signals.append(
                        Signal(
                            kind="breakout",
                            level=zone.level,
                            timestamp=latest.timestamp,
                            details="Bearish breakout below support",
                        )
                    )
        return signals

    def detect_reversals(self, bars: List[PriceBar], zones: Iterable[SupportResistanceZone]) -> List[Signal]:
        if len(bars) < 2:
            return []
        previous = bars[-2]
        latest = bars[-1]
        signals: List[Signal] = []
        for zone in zones:
            proximity = abs(latest.close - zone.level) / zone.level
            if proximity > self.zone_tolerance:
                continue
            if zone.kind in {"support", "mixed"}:
                if previous.close < previous.open and latest.close > latest.open:
                    signals.append(
                        Signal(
                            kind="reversal",
                            level=zone.level,
                            timestamp=latest.timestamp,
                            details="Bullish reversal near support",
                        )
                    )
            if zone.kind in {"resistance", "mixed"}:
                if previous.close > previous.open and latest.close < latest.open:
                    signals.append(
                        Signal(
                            kind="reversal",
                            level=zone.level,
                            timestamp=latest.timestamp,
                            details="Bearish reversal near resistance",
                        )
                    )
        return signals


def _true_ranges(bars: Sequence[PriceBar]) -> List[float]:
    if len(bars) < 2:
        return []
    ranges: List[float] = []
    for idx in range(1, len(bars)):
        current = bars[idx]
        previous = bars[idx - 1]
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return ranges


def _atr(bars: Sequence[PriceBar], period: int) -> Optional[float]:
    if len(bars) <= period:
        return None
    ranges = _true_ranges(bars[-(period + 1) :])
    if len(ranges) < period:
        return None
    return sum(ranges[-period:]) / period


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    smoothing = 2 / (period + 1)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value - ema_values[-1]) * smoothing + ema_values[-1])
    return ema_values


def _adx(bars: Sequence[PriceBar], period: int) -> Optional[float]:
    if len(bars) <= period + 1:
        return None
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for idx in range(1, len(bars)):
        up_move = bars[idx].high - bars[idx - 1].high
        down_move = bars[idx - 1].low - bars[idx].low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    tr_list = _true_ranges(bars)
    if len(tr_list) < period:
        return None
    tr_sum = sum(tr_list[-period:])
    plus_di = 100 * (sum(plus_dm[-period:]) / tr_sum) if tr_sum else 0
    minus_di = 100 * (sum(minus_dm[-period:]) / tr_sum) if tr_sum else 0
    if plus_di + minus_di == 0:
        return 0
    return 100 * abs(plus_di - minus_di) / (plus_di + minus_di)


def _find_swings(bars: Sequence[PriceBar], lookback: int) -> List[tuple[str, PriceBar, int]]:
    swings: List[tuple[str, PriceBar, int]] = []
    for idx in range(lookback, len(bars) - lookback):
        window = bars[idx - lookback : idx + lookback + 1]
        current = bars[idx]
        max_high = max(bar.high for bar in window)
        min_low = min(bar.low for bar in window)
        if current.high >= max_high:
            swings.append(("high", current, idx))
        if current.low <= min_low:
            swings.append(("low", current, idx))
    return swings


class RangeSweepSignalEngine:
    def __init__(
        self,
        range_window: int = 60,
        atr_window: int = 14,
        atr_threshold: float = 0.002,
        min_touches: int = 2,
        touch_tolerance: float = 0.0015,
        max_range_pct: float = 0.015,
        min_range_pct: float = 0.002,
        ema_period: int = 20,
        ema_slope_threshold: float = 0.0005,
        adx_threshold: float = 20.0,
        sweep_atr_multiplier: float = 0.4,
        failure_candles: int = 5,
        bos_lookback: int = 3,
    ) -> None:
        self.range_window = range_window
        self.atr_window = atr_window
        self.atr_threshold = atr_threshold
        self.min_touches = min_touches
        self.touch_tolerance = touch_tolerance
        self.max_range_pct = max_range_pct
        self.min_range_pct = min_range_pct
        self.ema_period = ema_period
        self.ema_slope_threshold = ema_slope_threshold
        self.adx_threshold = adx_threshold
        self.sweep_atr_multiplier = sweep_atr_multiplier
        self.failure_candles = failure_candles
        self.bos_lookback = bos_lookback

    def detect_range(self, bars: Sequence[PriceBar]) -> Optional[RangeContext]:
        if len(bars) < self.range_window:
            return None
        window = bars[-self.range_window :]
        range_high = max(bar.high for bar in window)
        range_low = min(bar.low for bar in window)
        range_mid = (range_high + range_low) / 2
        range_width = range_high - range_low
        if range_width <= 0:
            return None
        range_pct = range_width / range_mid
        if range_pct > self.max_range_pct or range_pct < self.min_range_pct:
            return None
        atr = _atr(window, self.atr_window)
        if atr is None:
            return None
        atr_percent = atr / range_mid
        if atr_percent > self.atr_threshold:
            return None
        ema_values = _ema([bar.close for bar in window], self.ema_period)
        if len(ema_values) < 2:
            return None
        ema_slope = (ema_values[-1] - ema_values[-2]) / range_mid
        adx = _adx(window, self.atr_window)
        if adx is not None and adx > self.adx_threshold:
            return None
        if abs(ema_slope) > self.ema_slope_threshold:
            return None
        tolerance = range_mid * self.touch_tolerance
        touches_high = sum(1 for bar in window if abs(bar.high - range_high) <= tolerance)
        touches_low = sum(1 for bar in window if abs(bar.low - range_low) <= tolerance)
        if touches_high < self.min_touches or touches_low < self.min_touches:
            return None
        quality_score = min(100.0, (touches_high + touches_low) * 10 + (1 - atr_percent / self.atr_threshold) * 30)
        return RangeContext(
            high=range_high,
            low=range_low,
            mid=range_mid,
            start=window[0].timestamp,
            end=window[-1].timestamp,
            touches_high=touches_high,
            touches_low=touches_low,
            atr_percent=atr_percent,
            ema_slope=ema_slope,
            adx=adx,
            quality_score=quality_score,
        )

    def _detect_sweep(self, range_ctx: RangeContext, bars: Sequence[PriceBar]) -> Optional[dict]:
        atr = _atr(bars, self.atr_window)
        if atr is None:
            return None
        buffer = atr * self.sweep_atr_multiplier
        for idx in range(len(bars) - 1, -1, -1):
            bar = bars[idx]
            if bar.high > range_ctx.high + buffer:
                return {"side": "high", "extreme": bar.high, "index": idx, "timestamp": bar.timestamp, "buffer": buffer}
            if bar.low < range_ctx.low - buffer:
                return {"side": "low", "extreme": bar.low, "index": idx, "timestamp": bar.timestamp, "buffer": buffer}
        return None

    def _confirm_failure(self, sweep: dict, range_ctx: RangeContext, bars: Sequence[PriceBar]) -> Optional[int]:
        end_index = min(len(bars) - 1, sweep["index"] + self.failure_candles)
        for idx in range(sweep["index"] + 1, end_index + 1):
            bar = bars[idx]
            close_inside = range_ctx.low <= bar.close <= range_ctx.high
            if sweep["side"] == "high":
                rejection = bar.high > range_ctx.high and bar.close <= range_ctx.high
                no_extension = bar.high <= sweep["extreme"]
            else:
                rejection = bar.low < range_ctx.low and bar.close >= range_ctx.low
                no_extension = bar.low >= sweep["extreme"]
            if close_inside and (rejection or no_extension):
                return idx
        return None

    def _confirm_bos(
        self,
        sweep: dict,
        failure_index: int,
        bars: Sequence[PriceBar],
    ) -> Optional[tuple[str, int, float]]:
        swings = _find_swings(bars, self.bos_lookback)
        if sweep["side"] == "low":
            swing_highs = [s for s in swings if s[0] == "high" and s[2] < failure_index]
            if not swing_highs:
                return None
            last_swing = swing_highs[-1]
            for idx in range(failure_index + 1, len(bars)):
                if bars[idx].close > last_swing[1].high:
                    return ("bullish", idx, last_swing[1].high)
        else:
            swing_lows = [s for s in swings if s[0] == "low" and s[2] < failure_index]
            if not swing_lows:
                return None
            last_swing = swing_lows[-1]
            for idx in range(failure_index + 1, len(bars)):
                if bars[idx].close < last_swing[1].low:
                    return ("bearish", idx, last_swing[1].low)
        return None

    def generate_signal(
        self,
        range_bars: Sequence[PriceBar],
        exec_bars: Sequence[PriceBar],
    ) -> RangeSweepSignal:
        range_ctx = self.detect_range(range_bars)
        if not range_ctx:
            return RangeSweepSignal(
                signal="NONE",
                reason=["no_valid_range"],
                entry_price=None,
                sl_price=None,
                tp1=None,
                tp2=None,
                confidence_score=0,
                range_high=None,
                range_low=None,
                sweep_extreme=None,
                timestamp=None,
            )
        sweep = self._detect_sweep(range_ctx, exec_bars)
        if not sweep:
            return RangeSweepSignal(
                signal="NONE",
                reason=["range_detected", "no_sweep"],
                entry_price=None,
                sl_price=None,
                tp1=None,
                tp2=None,
                confidence_score=range_ctx.quality_score,
                range_high=range_ctx.high,
                range_low=range_ctx.low,
                sweep_extreme=None,
                timestamp=None,
            )
        failure_index = self._confirm_failure(sweep, range_ctx, exec_bars)
        if failure_index is None:
            return RangeSweepSignal(
                signal="NONE",
                reason=["range_detected", f"{sweep['side']}_sweep", "no_failure"],
                entry_price=None,
                sl_price=None,
                tp1=None,
                tp2=None,
                confidence_score=range_ctx.quality_score,
                range_high=range_ctx.high,
                range_low=range_ctx.low,
                sweep_extreme=sweep["extreme"],
                timestamp=sweep["timestamp"],
            )
        bos = self._confirm_bos(sweep, failure_index, exec_bars)
        if not bos:
            return RangeSweepSignal(
                signal="NONE",
                reason=["range_detected", f"{sweep['side']}_sweep", "failure_confirmed", "no_bos"],
                entry_price=None,
                sl_price=None,
                tp1=None,
                tp2=None,
                confidence_score=range_ctx.quality_score,
                range_high=range_ctx.high,
                range_low=range_ctx.low,
                sweep_extreme=sweep["extreme"],
                timestamp=exec_bars[failure_index].timestamp,
            )
        direction, bos_index, _ = bos
        entry = exec_bars[bos_index].close
        buffer = sweep["buffer"]
        if direction == "bullish":
            sl_price = sweep["extreme"] - buffer
            tp1 = range_ctx.high
            risk = entry - sl_price
            tp2 = entry + risk * 2 if risk > 0 else None
            signal_side = "LONG"
        else:
            sl_price = sweep["extreme"] + buffer
            tp1 = range_ctx.low
            risk = sl_price - entry
            tp2 = entry - risk * 2 if risk > 0 else None
            signal_side = "SHORT"
        confidence = min(100.0, range_ctx.quality_score + 30)
        return RangeSweepSignal(
            signal=signal_side,
            reason=["range_detected", f"{sweep['side']}_sweep", "failure_confirmed", f"{direction}_bos"],
            entry_price=entry,
            sl_price=sl_price,
            tp1=tp1,
            tp2=tp2,
            confidence_score=confidence,
            range_high=range_ctx.high,
            range_low=range_ctx.low,
            sweep_extreme=sweep["extreme"],
            timestamp=exec_bars[bos_index].timestamp,
        )


def _format_zone(zone: SupportResistanceZone) -> str:
    return f"{zone.kind:<10} level={zone.level:.5f} touches={zone.touches}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capital.com support/resistance bot")

    parser.add_argument("--mode", choices=["support_resistance", "range_sweep"], default="support_resistance")
    parser.add_argument("--resolution", default="MINUTE", help="Candle resolution (support/resistance)")
    parser.add_argument("--hours", type=int, default=6, help="Lookback window in hours (support/resistance)")
    parser.add_argument("--range-resolution", default="MINUTE_15", help="Range timeframe resolution")
    parser.add_argument("--exec-resolution", default="MINUTE", help="Execution timeframe resolution")
    parser.add_argument("--range-hours", type=int, default=24, help="Range lookback window in hours")
    parser.add_argument("--exec-hours", type=int, default=6, help="Execution lookback window in hours")

    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--zone-tolerance", type=float, default=0.002)
    parser.add_argument("--breakout-threshold", type=float, default=0.001)
    parser.add_argument("--range-window", type=int, default=60)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--atr-threshold", type=float, default=0.002)
    parser.add_argument("--min-touches", type=int, default=2)
    parser.add_argument("--touch-tolerance", type=float, default=0.0015)
    parser.add_argument("--max-range-pct", type=float, default=0.015)
    parser.add_argument("--min-range-pct", type=float, default=0.002)
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--ema-slope-threshold", type=float, default=0.0005)
    parser.add_argument("--adx-threshold", type=float, default=20.0)
    parser.add_argument("--sweep-atr-multiplier", type=float, default=0.4)
    parser.add_argument("--failure-candles", type=int, default=5)
    parser.add_argument("--bos-lookback", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("CAPITAL_API_KEY")
    identifier = os.environ.get("CAPITAL_IDENTIFIER")
    password = os.environ.get("CAPITAL_PASSWORD")
    if not api_key or not identifier or not password:
        raise SystemExit("Missing CAPITAL_API_KEY, CAPITAL_IDENTIFIER, or CAPITAL_PASSWORD env vars")

    if not args.epic and not args.show_accounts:
        raise SystemExit("At least one --epic is required unless --show-accounts is set.")

    base_url = DEMO_BASE_URL if args.demo else args.base_url
    client = CapitalComClient(api_key, identifier, password, base_url)
    client.login()

        if args.mode == "support_resistance":
            end = datetime.utcnow()
            start = end - timedelta(hours=args.hours)
            bars = client.get_price_history(epic, args.resolution, start, end)
            if not bars:
                raise SystemExit(f"No price data returned for {epic}")
            analyzer = SupportResistanceAnalyzer(
                lookback=args.lookback,
                zone_tolerance=args.zone_tolerance,
                breakout_threshold=args.breakout_threshold,
            )
            swings = analyzer.find_swings(bars)
            zones = analyzer.build_zones(swings)
            breakouts = analyzer.detect_breakouts(bars, zones)
            reversals = analyzer.detect_reversals(bars, zones)

            print(f"\n[{epic}] Zones:")
            for zone in zones:
                print(f"  - {_format_zone(zone)}")

            print(f"\n[{epic}] Signals:")
            for signal in breakouts + reversals:
                print(f"  - {signal.kind:<8} level={signal.level:.5f} time={signal.timestamp} {signal.details}")
        else:
            range_end = datetime.utcnow()
            range_start = range_end - timedelta(hours=args.range_hours)
            exec_end = range_end
            exec_start = exec_end - timedelta(hours=args.exec_hours)
            range_bars = client.get_price_history(epic, args.range_resolution, range_start, range_end)
            exec_bars = client.get_price_history(epic, args.exec_resolution, exec_start, exec_end)
            if not range_bars or not exec_bars:
                raise SystemExit(f"No price data returned for {epic}")
            engine = RangeSweepSignalEngine(
                range_window=args.range_window,
                atr_window=args.atr_window,
                atr_threshold=args.atr_threshold,
                min_touches=args.min_touches,
                touch_tolerance=args.touch_tolerance,
                max_range_pct=args.max_range_pct,
                min_range_pct=args.min_range_pct,
                ema_period=args.ema_period,
                ema_slope_threshold=args.ema_slope_threshold,
                adx_threshold=args.adx_threshold,
                sweep_atr_multiplier=args.sweep_atr_multiplier,
                failure_candles=args.failure_candles,
                bos_lookback=args.bos_lookback,
            )
            signal = engine.generate_signal(range_bars, exec_bars)
            print(f"\n[{epic}] Range Sweep Signal:")
            print(f"  signal={signal.signal} confidence={signal.confidence_score:.1f}")
            print(f"  reason={signal.reason}")
            print(
                "  entry={entry} sl={sl} tp1={tp1} tp2={tp2}".format(
                    entry=f"{signal.entry_price:.5f}" if signal.entry_price else "n/a",
                    sl=f"{signal.sl_price:.5f}" if signal.sl_price else "n/a",
                    tp1=f"{signal.tp1:.5f}" if signal.tp1 else "n/a",
                    tp2=f"{signal.tp2:.5f}" if signal.tp2 else "n/a",
                )
            )
            print(
                "  range_high={high} range_low={low} sweep_extreme={sweep} time={time}".format(
                    high=f"{signal.range_high:.5f}" if signal.range_high else "n/a",
                    low=f"{signal.range_low:.5f}" if signal.range_low else "n/a",
                    sweep=f"{signal.sweep_extreme:.5f}" if signal.sweep_extreme else "n/a",
                    time=signal.timestamp,
                )
            )


if __name__ == "__main__":
    main()
