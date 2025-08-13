import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
import websockets
import pandas as pd
import ssl, certifi
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from html import escape as html_escape

WS_URL = "wss://pumpportal.fun/api/data"
COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
JUP_PRICE_API = "https://price.jup.ag/v4/price?ids=SOL"

TELEGRAM_TOKEN = "7896377577:AAEY34tj4Kx29HfZgpfFLG72VHI0VOvW32U"
PUBLIC_GROUP_ID = "-1002213254622"
INSECURE_SSL = os.getenv("INSECURE_SSL", "") == "1"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (PumpFunBot/2.0; +https://example.com)",
    "Accept": "application/json",
}

STOPWORDS = {"no", "unknown", "description", "available", "n/a", "none", "token", "the", "a", "an"}
TG_MAX = 3800

def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        if not token or not chat_id:
            raise RuntimeError("Faltan TELEGRAM_TOKEN o PUBLIC_GROUP_ID.")
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        timeout = ClientTimeout(total=20, connect=5)
        connector = TCPConnector(ssl=False) if INSECURE_SSL else TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
        self.session = aiohttp.ClientSession(timeout=timeout, headers=DEFAULT_HEADERS, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def send_message(self, text: str, parse_mode: str | None = "HTML", buttons: list | None = None):
        if not self.session:
            raise RuntimeError("TelegramBot session not initialized")
        url = f"{self.base}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": buttons}
        async with self.session.post(url, json=payload, ssl=False if INSECURE_SSL else None) as resp:
            if resp.status != 200:
                body = await resp.text()
                logging.error(f"Telegram sendMessage failed: {resp.status} - {body[:200]}")

class DataFetcher:
    def __init__(self):
        columns = [
            "Name","Symbol","Timestamp","Sentiment","Sentiment Description",
            "Market Cap (SOL)","Market Cap (USD)","CA Address","Image URI",
            "Description","Supply","Price","Market Cap Change (SOL)",
            "telegram","trading_page"
        ]
        self.tokens_df = pd.DataFrame(columns=columns)
        self.session: aiohttp.ClientSession | None = None
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.sol_price: float = 0.0
        self._sol_price_ts: float = 0.0

    async def __aenter__(self):
        timeout = ClientTimeout(total=15, connect=5)
        connector = TCPConnector(ssl=False) if INSECURE_SSL else None
        self.session = aiohttp.ClientSession(timeout=timeout, headers=DEFAULT_HEADERS, connector=connector)
        self.websocket = await self._connect_ws()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

    async def _connect_ws(self):
        try:
            return await websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_queue=64,
                extra_headers={"User-Agent": DEFAULT_HEADERS["User-Agent"], "Origin": "https://pump.fun"},
            )
        except Exception as e:
            logging.error(f"WS connect failed: {e}")
            return None

    async def fetch_sol_price_coingecko(self) -> float:
        assert self.session is not None
        async with self.session.get(COINGECKO_API) as r:
            if r.status != 200:
                return 0.0
            data = await r.json()
            return float(data.get("solana", {}).get("usd", 0.0)) or 0.0

    async def fetch_sol_price_jup(self) -> float:
        assert self.session is not None
        async with self.session.get(JUP_PRICE_API) as r:
            if r.status != 200:
                return 0.0
            data = await r.json()
            return float(data.get("data", {}).get("SOL", {}).get("price", 0.0)) or 0.0

    async def fetch_sol_price_cached(self, ttl_sec: int = 120) -> float:
        now = time.time()
        if self.sol_price and (now - self._sol_price_ts) < ttl_sec:
            return self.sol_price
        p = 0.0
        try:
            p = await self.fetch_sol_price_coingecko()
            if p > 0:
                logging.info(f"SOL price (CoinGecko): {p}")
        except Exception as e:
            logging.warning(f"CoinGecko price failed: {e}")
        if p == 0.0:
            try:
                p = await self.fetch_sol_price_jup()
                if p > 0:
                    logging.info(f"SOL price (Jupiter): {p}")
            except Exception as e:
                logging.warning(f"Jupiter price failed: {e}")
        if p > 0:
            self.sol_price = p
            self._sol_price_ts = now
            return p
        if self.sol_price:
            logging.warning("Using last known SOL price due to refresh failure.")
            return self.sol_price
        logging.error("SOL price unavailable.")
        return 0.0

    async def subscribe_new_tokens(self):
        if not self.websocket:
            self.websocket = await self._connect_ws()
            if not self.websocket:
                return None
        try:
            await self.websocket.send(json.dumps({"method": "subscribeNewToken"}))
            logging.info("Subscribed to new tokens.")
            return True
        except Exception as e:
            logging.error(f"subscribeNewToken failed: {e}")
            return None

    @staticmethod
    def _to_float(x):
        try:
            if isinstance(x, str):
                x = x.replace(",", "")
            return float(x)
        except Exception:
            return None

    async def next_new_token(self, wait_sec: int = 20) -> Dict | None:
        if not self.websocket:
            self.websocket = await self._connect_ws()
            if not self.websocket:
                return None
            await self.subscribe_new_tokens()
        try:
            raw = await asyncio.wait_for(self.websocket.recv(), timeout=wait_sec)
        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WS closed, reconnecting...")
            self.websocket = await self._connect_ws()
            if self.websocket:
                await self.subscribe_new_tokens()
            return None
        except Exception as e:
            logging.error(f"WS recv error: {e}")
            return None
        try:
            msg = json.loads(raw)
        except Exception:
            logging.warning("WS message not JSON; ignoring.")
            return None
        d = msg.get("data") or msg
        token = d.get("token") or d
        name = token.get("name") or "Unknown"
        symbol = token.get("symbol") or "Unknown"
        ts = token.get("timestamp") or d.get("timestamp") or 0
        mint = token.get("mint") or token.get("address") or "Unknown"
        img = token.get("image_uri") or token.get("image") or ""
        desc = token.get("description") or ""
        mc_sol = self._to_float(token.get("market_cap") or token.get("vSolInBondingCurve"))
        mc_usd = self._to_float(token.get("usd_market_cap"))
        if mc_usd is None and mc_sol is not None and self.sol_price:
            mc_usd = mc_sol * self.sol_price
        if mc_sol is None and mc_usd is not None and self.sol_price:
            mc_sol = mc_usd / self.sol_price
        return {
            "Name": name,
            "Symbol": symbol,
            "Timestamp": self.format_timestamp(ts),
            "Market Cap (SOL)": mc_sol,
            "Market Cap (USD)": mc_usd,
            "CA Address": mint,
            "Image URI": img,
            "Description": desc,
            "telegram": "https://t.me/pumpfun",
            "trading_page": f"https://pump.fun/trades/{mint}" if mint and mint != "Unknown" else "https://pump.fun",
            "Supply": None,
            "Price": None,
        }

    @staticmethod
    def format_timestamp(timestamp: int) -> str:
        try:
            ts = timestamp / 1000 if timestamp and timestamp > 10**12 else (timestamp or time.time())
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def update_tokens_df(self, token_data: Dict):
        row = {col: token_data.get(col) for col in self.tokens_df.columns}
        self.tokens_df.loc[len(self.tokens_df)] = row
        self.tokens_df.drop_duplicates(subset=["CA Address"], keep="last", inplace=True)
        self.tokens_df["Market Cap (SOL)"] = pd.to_numeric(self.tokens_df["Market Cap (SOL)"], errors="coerce")
        self.tokens_df["Market Cap Change (SOL)"] = (
            self.tokens_df.groupby("CA Address", dropna=False)["Market Cap (SOL)"].diff()
        )

class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(text: str):
        try:
            if not text or not isinstance(text, str):
                return 0.0, "Unknown"
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.2:
                desc = "Positive 😊"
            elif polarity < -0.2:
                desc = "Negative 😢"
            else:
                desc = "Neutral 😐"
            return polarity, desc
        except Exception as e:
            logging.error(f"Sentiment error: {e}")
            return 0.0, "Unknown"

    def analyze(self, token_data: Dict, rel_growth: float) -> Dict:
        s, sdesc = self.analyze_sentiment(token_data.get("Description", ""))
        token_data["Sentiment"] = s
        token_data["Sentiment Description"] = sdesc
        market_sentiment = 0.6 if rel_growth > 0.1 else (-0.6 if rel_growth < -0.1 else 0.0)
        token_data["Combined Sentiment"] = 0.7 * s + 0.3 * market_sentiment
        return token_data

class TrendAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, df: pd.DataFrame) -> Dict[str, str]:
        try:
            return {"overall_trends": self.analyze_trends(df), "keyword_insights": self.analyze_trends_from_descriptions(df)}
        except Exception as e:
            self.logger.error(f"Trend analyze error: {e}")
            return {"overall_trends": "Error.", "keyword_insights": "Error."}

    def analyze_trends(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No data available for analysis."
        try:
            capdf = df[df["Market Cap (SOL)"].notna()].copy()
            if capdf.empty:
                return "No market-cap data yet."
            top_tokens = capdf.nlargest(5, "Market Cap (SOL)")
            top_tokens_details = [
                f"{row['Symbol']}: {row['Name']} - {row['Market Cap (SOL)']:.2f} SOL\n  Contract: {row['CA Address']}\n  Price: {row.get('Price') or 'N/A'} SOL\n  Supply: {row.get('Supply') or 'N/A'}"
                for _, row in top_tokens.iterrows()
            ]
            high_cap_avg = float(capdf["Market Cap (SOL)"].mean())
            high_cap_count = int(len(capdf))
            capdf["Prev Cap"] = capdf.groupby("CA Address")["Market Cap (SOL)"].shift(1)
            capdf["Rel Growth"] = (capdf["Market Cap (SOL)"] - capdf["Prev Cap"]) / (capdf["Prev Cap"] + 1e-9)
            sig = capdf[(capdf["Prev Cap"].notna()) & (capdf["Rel Growth"] > 0.10)]
            sig_details = [
                f"{row['Symbol']}: {row['Name']} - Growth: {row['Rel Growth']*100:.1f}%\n  Contract: {row['CA Address']}\n  Price: {row.get('Price') or 'N/A'} SOL\n  Supply: {row.get('Supply') or 'N/A'}"
                for _, row in sig.iterrows()
            ]
            sentiment_counts = df.get("Sentiment Description", pd.Series([], dtype=str)).value_counts().to_dict()
            sentiment_examples = {}
            for k in sentiment_counts.keys():
                subset = df[(df["Sentiment Description"] == k) & (df["Symbol"].notna()) & (df["Symbol"] != "Unknown")]
                sentiment_examples[k] = ", ".join([f"{row['Symbol']}" for _, row in subset.head(3).iterrows()]) or "N/A"
            s = "Current Trends on Pump.Fun\n"
            if top_tokens_details:
                s += "- Top Tokens by Market Cap:\n" + "\n".join([f"  - {d}" for d in top_tokens_details]) + "\n"
            s += f"- High Cap Tokens: {high_cap_count} tokens\n"
            s += f"  - Average Market Cap: {high_cap_avg:.2f} SOL\n"
            if sentiment_counts:
                s += "Sentiment Overview:\n"
                for sentiment, count in sentiment_counts.items():
                    s += f"- {sentiment}: {count} tokens (e.g., {sentiment_examples.get(sentiment,'N/A')})\n"
            s += "- Significant Growth:\n"
            s += ("\n".join([f"  - {d}" for d in sig_details]) + "\n") if sig_details else "  - None.\n"
            return s
        except Exception as e:
            self.logger.error(f"analyze_trends error: {e}")
            return "Error analyzing trends."

    def analyze_trends_from_descriptions(self, df: pd.DataFrame) -> str:
        if df.empty or "Description" not in df.columns:
            return "No data available for description-based analysis."
        try:
            descriptions = df["Description"].dropna().astype(str).str.lower()
            descriptions = descriptions[descriptions.str.strip() != ""]
            if descriptions.empty:
                return ""
            cv = CountVectorizer(max_features=10)
            cv.fit([" ".join(descriptions.tolist())])
            keywords = [k for k in cv.get_feature_names_out().tolist() if k not in STOPWORDS]
            if not keywords:
                return ""
            keyword_counter = Counter()
            keyword_to_tokens = {k: [] for k in keywords}
            for keyword in keywords:
                mask = df["Description"].astype(str).str.lower().str.contains(keyword, regex=False, na=False)
                matches = df[mask]
                keyword_counter[keyword] = len(matches)
                keyword_to_tokens[keyword] = matches["Symbol"].dropna().tolist()
            blocks = []
            for keyword in keywords:
                tokens = df[df["Symbol"].isin(keyword_to_tokens[keyword])]
                avg_market_cap = float(tokens["Market Cap (SOL)"].mean() or 0)
                avg_sentiment = float(tokens.get("Sentiment", pd.Series([0] * len(tokens))).mean() or 0)
                top_tokens = tokens.nlargest(3, "Market Cap (SOL)")[["Symbol", "Market Cap (SOL)"]] if not tokens.empty else tokens
                top_tokens_str = ", ".join([f"{row['Symbol']} ({row['Market Cap (SOL)']:.2f} SOL)" for _, row in top_tokens.iterrows()]) if not top_tokens.empty else ""
                blocks.append(
                    f"Keyword: {keyword}\n   - Occurrences: {keyword_counter[keyword]}\n   - Avg Market Cap: {avg_market_cap:.2f} SOL\n   - Avg Sentiment: {avg_sentiment:.2f}\n   - Top Tokens: {top_tokens_str if top_tokens_str else 'None'}"
                )
            return "Contextual Keyword Insights\n" + "\n\n".join(blocks)
        except Exception as e:
            self.logger.error(f"analyze_trends_from_descriptions error: {e}")
            return ""

class MessageFormatter:
    @staticmethod
    def format_latest_token(token_data: dict, sol_price: float) -> str:
        mc_sol = token_data.get("Market Cap (SOL)")
        mc_usd = token_data.get("Market Cap (USD)")
        if mc_usd is None and (mc_sol is not None) and sol_price:
            mc_usd = mc_sol * sol_price
        name = html_escape(str(token_data.get("Name", "")))
        sym = html_escape(str(token_data.get("Symbol", ""))).upper()[:16]
        ca = html_escape(str(token_data.get("CA Address", "")))
        trading_page_link = token_data.get("trading_page", f"https://pump.fun/trades/{token_data.get('CA Address','')}")
        mc_sol_str = f"{mc_sol:,.2f} SOL" if isinstance(mc_sol, (int, float)) else "N/A"
        mc_usd_str = f"${mc_usd:,.2f}" if isinstance(mc_usd, (int, float)) else "N/A"
        return (
            "🚀 <b>Latest Token on Pump.Fun</b> 🚀\n"
            f"🌟 <b>Name</b>: <a href=\"{trading_page_link}\">{name}</a>\n"
            f"💎 <b>Symbol</b>: <a href=\"{trading_page_link}\">{sym}</a>\n"
            f"📜 <b>Contract Address</b>: <code>{ca}</code>\n"
            f"📈 <b>Market Cap</b>: <b>{mc_sol_str}</b> (~{mc_usd_str} USDT)\n"
        )

def _chunk(s: str, n: int = TG_MAX):
    for i in range(0, len(s), n):
        yield s[i:i+n]

async def main():
    configure_logging()
    if not TELEGRAM_TOKEN or not PUBLIC_GROUP_ID:
        raise RuntimeError("Faltan TELEGRAM_TOKEN o PUBLIC_GROUP_ID.")
    sentiment_analyzer = SentimentAnalyzer()
    trend_analyzer = TrendAnalyzer()
    formatter = MessageFormatter()
    async with DataFetcher() as fetcher, TelegramBot(TELEGRAM_TOKEN, PUBLIC_GROUP_ID) as tg:
        await fetcher.subscribe_new_tokens()
        while True:
            try:
                fetcher.sol_price = await fetcher.fetch_sol_price_cached()
                if fetcher.sol_price == 0.0:
                    await asyncio.sleep(20)
                    continue
                token_data = await fetcher.next_new_token(wait_sec=25)
                if not token_data:
                    await asyncio.sleep(10)
                    continue
                rel = 0.0
                try:
                    ca = token_data.get("CA Address")
                    mc_now = token_data.get("Market Cap (SOL)")
                    prev = fetcher.tokens_df.loc[fetcher.tokens_df["CA Address"] == ca, "Market Cap (SOL)"].dropna().tail(1)
                    if mc_now is not None and len(prev) == 1:
                        pv = float(prev.iloc[0]); cv = float(mc_now)
                        rel = (cv - pv) / (pv + 1e-9)
                except Exception:
                    rel = 0.0
                token_data = sentiment_analyzer.analyze(token_data, rel)
                fetcher.update_tokens_df(token_data)
                trends = trend_analyzer.analyze(fetcher.tokens_df)
                body = trends["overall_trends"]
                ki = trends["keyword_insights"]
                if ki:
                    body += "\n\n" + ki
                for part in _chunk(body):
                    await tg.send_message(part, parse_mode=None)
                ca = token_data.get("CA Address") or ""
                trading_page_link = token_data.get("trading_page")
                dex_url = f"https://dexscreener.com/solana/{ca}" if ca else "https://dexscreener.com/solana"
                buttons = [
                    [{"text": "Trading en Pump.Fun", "url": trading_page_link}],
                    [{"text": "Ver en Dexscreener", "url": dex_url}],
                ]
                latest_token_msg = formatter.format_latest_token(token_data, fetcher.sol_price)
                await tg.send_message(latest_token_msg, parse_mode="HTML", buttons=buttons)
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Unexpected error in loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal error: {e}")
