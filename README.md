# UK Gilt Biography Agent

`uk_bond_biography_agent.py` generates a "bond biography" chapter, as a Jupyter notebook, for any of the 502 British government stocks in `UK_Gilts_Bond_Database.xlsx`. It is the UK counterpart of the Hall-Payne-Sargent US bond biography generator.

## Requirements

```bash
pip install pandas numpy matplotlib openpyxl nbformat nbconvert ipykernel
```

Keep `uk_bond_biography_agent.py` and `UK_Gilts_Bond_Database.xlsx` in the same folder. Nothing else is needed.

## What is in this repository

| File | What it is |
|---|---|
| `uk_bond_biography_agent.py` | The generator |
| `UK_Gilts_Bond_Database.xlsx` | The database it reads: BondList, BondPrice, BondQuant, RPI and a log of corrections |
| `chapters/` | Sample chapters |

The database was assembled from the Ellison-Scott UK gilt database (BGSDetails, BGSAmounts, BGSPrices and the earlier Amounts_RE / Prices_RE / Specification_RE files). Those source workbooks are not redistributed here; the `Corrections` sheet records every change made to them.

## Quick start

```bash
cd "Ellison-Scott-update Generator"
python uk_bond_biography_agent.py --search "War Loan"            # find the L1 ID
python uk_bond_biography_agent.py --bond-id 32400 --execute      # write and run the chapter
```

The chapter is saved to `chapters/chapter_32400_3_1-2pct_War_Loan.ipynb`.

## Commands

| Command | What it does |
|---|---|
| `--bond-id 32400 [20100 ...]` | Generate one or more chapters |
| `--search "Consols"` | Search by name or type (add `--generate-first` to generate the first match) |
| `--list-categories` | Stock types with counts |
| `--list-candidates 30` | The 30 stocks with the richest material (price history, size, special features) |
| `--all [--min-months 120]` | Every stock except tranches, optionally only those with at least N monthly prices |
| `--interactive` | Choose a stock interactively |
| `--execute` | Run each notebook after writing it, so the charts are saved in it |
| `--print-prompt` | Also print a ready-to-use prompt for turning the draft into a finished chapter |
| `--output-dir DIR` | Output folder (default: `chapters/` next to the script) |
| `--output NAME.ipynb` | File name (single stock only) |
| `--data PATH` | Use a database file stored elsewhere |

Good first examples:

| ID | Stock | Type |
|---|---|---|
| 32400 | 3½% War Loan | Undated |
| 20100 | 8¾% Treasury 1997 | Enlarged by tranches |
| 8400 | 9¼% Exchequer 1982 | Partly paid |
| 13300 | 3% British Transport 1978-88 | Nationalisation stock, double-dated |
| 51700 | 2½% IL 2024 | Old-style index-linked |
| 55500 | 1¼% IL 2055 | First new-style index-linked |
| 6200 | 9% Treasury Convertible 1980 | Convertible |
| 22100 | Floating Rate Treasury 1999 | Floating rate |
| 32202 | 0⅞% Green Gilt 2033 | Green gilt |

## What a chapter contains

A chapter has 18 to 30 cells, depending on the data available:

1. Title, setup cells and an "At a Glance" table.
2. Overview.
3. The Instrument: terms, plus plain-English explanations of UK features such as double-dated, undated, partly paid, convertible, index-linked, tax effects and death-duty stocks.
4. Historical Context: policy eras, prime ministers, and events with this stock's price and yield response.
5. Lifecycle timeline.
6. Issuance and amount outstanding: taps, tranche amalgamations and reductions detected automatically; market value.
7. Market price, broken down by era.
8. Yield and relative value against peers of similar remaining life.
9. The stock on the yield curve, with a fitted Nelson-Siegel curve.
10. Nominal and real total returns, duration, volatility and drawdown.
11. Market events: unusual months, with the nearest historical event.
12. Related stocks.
13. Issuance, distribution and redemption.
14. Implications and legacy, then data notes and references.

Places that need hand-written narrative are marked `<!-- ENRICH: ... -->`, about 13 per chapter.

Every notebook is self-contained, because the helper functions are embedded in it. It finds the database in its own folder or a parent folder. To store the database elsewhere, set `UK_GILTS_DB=/path/to/UK_Gilts_Bond_Database.xlsx`.

## Enrichment workflow

1. Generate the draft with `--execute --print-prompt`.
2. Give the printed prompt to Claude (or edit by hand). The prompt asks for every ENRICH marker to be replaced and the result saved to `chapters/enhanced/..._enhanced.ipynb`.
3. Verify the enhanced chapter:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace chapters/enhanced/<file>.ipynb
   ```

## Conventions and caveats

**Prices** are month-end clean prices per £100 nominal.

- Before 28 February 1986, the source quotes stocks with more than 5 years to maturity, and undated stocks, including accrued interest ("dirty" prices).
- The agent converts these to clean prices before computing yields and returns. `clean_price()` does the conversion; `series(db, 'price', id, 'Average')` returns the raw source prices.

**Yields:**

| Stock | Yield measure |
|---|---|
| Conventional | Gross redemption yield. A double-dated stock is measured to its earliest date when priced at or above par, otherwise to its final date. |
| Undated | Flat yield (coupon ÷ price), up to the date its redemption was announced. |
| Index-linked (new style, 3-month lag) | Real yield from the real price. |
| Index-linked (old style, 8-month lag) | Real yield assuming 3% future RPI inflation. |
| Variable or floating rate | No yield. |

Yields within 3 months of redemption are dropped.

**Peers:**

| Stock | Peer group |
|---|---|
| Conventional | Conventional gilts whose remaining life is within max(1 year, 15%) of this stock's; at least 3 peers |
| Index-linked | Index-linked gilts within max(3 years, 35%); at least 2 peers |
| Undated | Conventional gilts with at least 15 years to redemption |

**Detection thresholds:**

- **Market events:** a price move of at least 5%, or a yield move at least 40 bp larger than the peer median's (the 12 largest are shown).
- **Large amount changes:** at least £50m and at least 5% of the amount outstanding.

**Knowledge base:** edit `UK_EVENTS`, `GOVERNMENTS`, `ERAS`, `LITERATURE` and `BOND_NOTES` near the top of the script. Check event dates marked `month=True` before publication, because only the month is certain for those.
