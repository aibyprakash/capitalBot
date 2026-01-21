"""Capital.com intraday support/resistance bot.

This module fetches intraday price history from Capital.com and identifies
support/resistance zones, along with breakout and reversal signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import os
from typing import Iterable, List, Optional

import requests


def _parse_iso_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


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
            bars.append(
                PriceBar(
                    timestamp=_parse_iso_time(bar["snapshotTimeUTC"]),
                    open=float(bar["openPrice"]["bid"]),
                    high=float(bar["highPrice"]["bid"]),
                    low=float(bar["lowPrice"]["bid"]),
                    close=float(bar["closePrice"]["bid"]),
                )
            )
        return bars


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


def _format_zone(zone: SupportResistanceZone) -> str:
    return f"{zone.kind:<10} level={zone.level:.5f} touches={zone.touches}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capital.com support/resistance bot")
    parser.add_argument("--epic", required=True, help="Market epic (e.g., FX.EURUSD)")
    parser.add_argument("--resolution", default="MINUTE", help="Candle resolution")
    parser.add_argument("--hours", type=int, default=6, help="Lookback window in hours")
    parser.add_argument("--base-url", default="https://api-capital.backend-capital.com")
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--zone-tolerance", type=float, default=0.002)
    parser.add_argument("--breakout-threshold", type=float, default=0.001)
    args = parser.parse_args()

    api_key = os.environ.get("CAPITAL_API_KEY")
    identifier = os.environ.get("CAPITAL_IDENTIFIER")
    password = os.environ.get("CAPITAL_PASSWORD")
    if not api_key or not identifier or not password:
        raise SystemExit("Missing CAPITAL_API_KEY, CAPITAL_IDENTIFIER, or CAPITAL_PASSWORD env vars")

    client = CapitalComClient(api_key, identifier, password, args.base_url)
    client.login()

    end = datetime.utcnow()
    start = end - timedelta(hours=args.hours)
    bars = client.get_price_history(args.epic, args.resolution, start, end)
    if not bars:
        raise SystemExit("No price data returned")

    analyzer = SupportResistanceAnalyzer(
        lookback=args.lookback,
        zone_tolerance=args.zone_tolerance,
        breakout_threshold=args.breakout_threshold,
    )
    swings = analyzer.find_swings(bars)
    zones = analyzer.build_zones(swings)
    breakouts = analyzer.detect_breakouts(bars, zones)
    reversals = analyzer.detect_reversals(bars, zones)

    print("\nZones:")
    for zone in zones:
        print(f"  - {_format_zone(zone)}")

    print("\nSignals:")
    for signal in breakouts + reversals:
        print(f"  - {signal.kind:<8} level={signal.level:.5f} time={signal.timestamp} {signal.details}")


if __name__ == "__main__":
    main()
