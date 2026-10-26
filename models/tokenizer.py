"""Word tokenizer for the RNN/CNN models (TextCNN, BiLSTM-Attention).

Lowercases text for GloVe 6B (uncased) and maps three token types to placeholders:
  <url>    http(s):// or www. links;
  <ticker> any $-prefixed symbol ($AAPL, $BRK.B), or a bare uppercase symbol from LARGE_CAPS;
  <num>    numbers such as 5, 2026, 1,200 or 3.5.
A bare uppercase word that is not in LARGE_CAPS stays a normal word, so acronyms like
AI, IT, US, CEO, GDP, ON, ALL are never tickers (SPEC.md). Fit nothing here: the vocabulary
is built separately, on train only.
"""
import html
import re
import unicodedata

URL, TICKER, NUM = "<url>", "<ticker>", "<num>"
SPECIAL_TOKENS = (URL, TICKER, NUM)

# Large, frequently quoted US symbols. Single letters, English words (NOW, LOW, KEY, CAT,
# ALL, ON...) and symbols with a common other meaning in FNS (MA = moving average, BA =
# exchange suffix as in GGAL.BA) are left out on purpose: a bare match must be unambiguous.
LARGE_CAPS = frozenset("""
AAPL MSFT AMZN GOOGL GOOG META TSLA NVDA AVGO ORCL ADBE CRM INTC AMD CSCO QCOM TXN IBM MU
NFLX DIS CMCSA CHTR TMUS VZ JPM BAC WFC GS AXP PYPL JNJ PFE MRK ABBV LLY UNH CVS AMGN
GILD BMY MRNA WMT COST TGT HD MCD SBUX NKE KO PEP PG XOM CVX COP OXY GE GM LMT RTX HON
MMM UPS FDX UBER LYFT ABNB SHOP SNAP PINS ZM PLTR COIN RIVN LCID NIO BABA TSM AMC GME SPY
QQQ IWM DIA
""".split())

# Never tickers when written without '$' (SPEC.md); kept explicit so the rule is testable.
NOT_TICKERS = frozenset("AI IT US CEO GDP ON ALL".split())
assert not LARGE_CAPS & NOT_TICKERS

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOLLAR_TICKER_RE = re.compile(r"\$[A-Za-z]{1,6}(?:\.[A-Za-z]{1,2})?\b")
# A bare symbol must stand alone: not glued to '&' or '.' (PG&E, GGAL.BA) or other letters.
BARE_CAPS_RE = re.compile(r"(?<![\w&.$])[A-Z]{2,5}(?![\w&]|\.\w)")
TOKEN_RE = re.compile(r"<url>|<ticker>|[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.,]\d+)*|[^\w\s]|_")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
CLITIC_RE = re.compile(r"(.+?)(n't|'s|'re|'ll|'ve|'d|'m)")


def normalize(text):
    """Return the text with HTML entities decoded, Unicode NFKC-normalized and URLs/tickers replaced.

    Case is kept (bare tickers are detected on uppercase); lowercasing happens in tokenize().
    """
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'").replace("‘", "'")
    text = URL_RE.sub(f" {URL} ", text)
    text = DOLLAR_TICKER_RE.sub(f" {TICKER} ", text)
    return BARE_CAPS_RE.sub(lambda m: f" {TICKER} " if m.group() in LARGE_CAPS else m.group(), text)


def tokenize(text):
    """Split one headline into lowercase tokens.

    Input: raw text. Output: list of tokens; clitics split as in GloVe's Stanford tokenizer
    ("apple's" -> apple 's, "don't" -> do n't), other inner apostrophes kept ("o'neill"),
    each punctuation mark is its own token, numbers become <num>.
    """
    tokens = []
    for t in TOKEN_RE.findall(normalize(text).lower()):
        if NUMBER_RE.fullmatch(t):
            tokens.append(NUM)
        elif m := CLITIC_RE.fullmatch(t):
            tokens.extend(m.groups())
        else:
            tokens.append(t)
    return tokens
