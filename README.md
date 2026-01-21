# Capital.com Support/Resistance Bot

This repository provides a lightweight intraday trading assistant that uses the Capital.com REST API to
identify support and resistance zones, then flags breakout and trend reversal signals.

## Features

- Logs into Capital.com with your API key and account credentials.
- Fetches intraday price history for a market epic.
- Finds swing highs/lows and clusters them into support/resistance zones.
- Detects breakouts beyond zones and reversal patterns near zones.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Export your credentials (do not hard-code them):

```bash
export CAPITAL_API_KEY="your_api_key"
export CAPITAL_IDENTIFIER="your_email_or_username"
export CAPITAL_PASSWORD="your_password"
```

Optionally, verify the exports before running the bot:

```bash
python - <<'PY'
import os
missing = [key for key in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD") if not os.getenv(key)]
print("Missing:", ", ".join(missing) if missing else "none")
PY
```

3. Run the bot:

```bash
python capital_bot.py --epic FX.EURUSD --resolution MINUTE --hours 6
```

To verify credentials and fetch your account balances (including demo, if your credentials point to a demo account):

```bash
python capital_bot.py --epic FX.EURUSD --show-accounts
```

To connect to the demo API, add `--demo`:

```bash
python capital_bot.py --demo --show-accounts
```

## Configuration

The CLI accepts parameters to tune the analysis:

- `--lookback`: window size for swing detection.
- `--zone-tolerance`: percent tolerance for clustering zones (e.g., `0.002` = 0.2%).
- `--breakout-threshold`: percent threshold beyond a zone to confirm breakout.
- `--mode`: switch between `support_resistance` and `range_sweep`.

### Range sweep mode

Use range + execution timeframes to detect range/sweep/BOS signals:

```bash
python capital_bot.py \
  --mode range_sweep \
  --epic METALS.GOLD \
  --epic ENERGY.CRUDE \
  --range-resolution MINUTE_15 \
  --exec-resolution MINUTE \
  --range-hours 24 \
  --exec-hours 6
```

## Notes

- Ensure your Capital.com account has API access enabled.
- This tool is for educational purposes and does not constitute financial advice.
