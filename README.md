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

3. Run the bot:

```bash
python capital_bot.py --epic FX.EURUSD --resolution MINUTE --hours 6
```

## Configuration

The CLI accepts parameters to tune the analysis:

- `--lookback`: window size for swing detection.
- `--zone-tolerance`: percent tolerance for clustering zones (e.g., `0.002` = 0.2%).
- `--breakout-threshold`: percent threshold beyond a zone to confirm breakout.

## Notes

- Ensure your Capital.com account has API access enabled.
- This tool is for educational purposes and does not constitute financial advice.
