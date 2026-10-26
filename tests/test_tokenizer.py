"""Tests for models/tokenizer.py."""
import unittest

from models.tokenizer import LARGE_CAPS, NOT_TICKERS, NUM, TICKER, URL, tokenize


class TickerTest(unittest.TestCase):
    def test_dollar_symbols_are_always_tickers(self):
        self.assertEqual(tokenize("$AAPL up, $BRK.B flat, $AI jumps"),
                         [TICKER, "up", ",", TICKER, "flat", ",", TICKER, "jumps"])

    def test_spec_acronyms_are_never_bare_tickers(self):
        tokens = tokenize("AI IT US CEO GDP ON ALL")
        self.assertNotIn(TICKER, tokens)
        self.assertEqual(tokens, ["ai", "it", "us", "ceo", "gdp", "on", "all"])
        self.assertFalse(LARGE_CAPS & NOT_TICKERS)

    def test_bare_large_cap_only_when_uppercase(self):
        self.assertEqual(tokenize("TSLA beats; Tesla and tsla fans cheer"),
                         [TICKER, "beats", ";", "tesla", "and", "tsla", "fans", "cheer"])

    def test_symbols_glued_to_ampersand_or_dot_are_not_tickers(self):
        self.assertNotIn(TICKER, tokenize("PG&E bankruptcy plan; GGAL.BA call; 200 MA cross"))
        self.assertEqual(tokenize("GM, PG."), [TICKER, ",", TICKER, "."])

    def test_dollar_amount_is_not_a_ticker(self):
        self.assertEqual(tokenize("Raises $5.2 billion"), ["raises", "$", NUM, "billion"])


class OtherTokensTest(unittest.TestCase):
    def test_urls_numbers_and_punctuation(self):
        self.assertEqual(tokenize("$BYND - JPMorgan cuts view, EPS 1,200% https://t.co/bd0xbFGjkT"),
                         [TICKER, "-", "jpmorgan", "cuts", "view", ",", "eps", NUM, "%", URL])

    def test_apostrophes_accents_and_html_entities(self):
        self.assertEqual(tokenize("Nestlé’s Q3 &amp; don't"),
                         ["nestlé", "'s", "q", NUM, "&", "do", "n't"])

    def test_clitics_split_like_glove_other_apostrophes_kept(self):
        self.assertEqual(tokenize("Investors' fears: we'll see O'Neill's plan, can't stop"),
                         ["investors", "'", "fears", ":", "we", "'ll", "see", "o'neill", "'s",
                          "plan", ",", "ca", "n't", "stop"])

    def test_empty_text(self):
        self.assertEqual(tokenize("   "), [])


if __name__ == "__main__":
    unittest.main()
