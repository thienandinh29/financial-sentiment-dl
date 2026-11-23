# LLM-made annotations — NOT gold

Every label file in this folder was produced with an LLM, not by a person reading the headline.
The team found this on 23/11/2026 (see `context/decisions.md`). They are kept only for honesty
and for one later comparison; they are **not** gold labels.

| File | What it is |
| --- | --- |
| `trial/tv1.csv`, `trial/tv2.csv`, `trial/tv3.csv` | Trial-30 labels of all three members (LLM-made) |
| `trial_agreement.json` | Agreement and kappa computed from those LLM files (not human agreement) |
| `trial_adjudication.csv` | Conclusions for the five trial disagreements, based on the LLM files |
| `round1/tv3.csv` | TV3's round-1 labels for 68 test headlines (LLM-made) |

TV1's round-1 labels (69 + 65) were also LLM-made; they were never committed and must not be.

Rules for these files:

- Never use them as gold, for kappa, to adjudicate, or as training data (the round-1 IDs are test headlines).
- Do not open them while hand-labeling, so they cannot anchor the human labels.
- Do not compare them with the human gold before week 6. On test headlines they are LLM
  predictions, so comparing them early would be a test evaluation (AGENTS.md rule 1).
  In week 6 they may be compared once with the locked gold and reported as
  "LLM labels vs human gold".

Human gold labels go to `labeling/gold/` (columns `id,label,reason`).
