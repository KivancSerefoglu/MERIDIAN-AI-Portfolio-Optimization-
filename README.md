# MERIDIAN — Portfolio Intelligence

> *Exposing what you're actually holding.*

MERIDIAN is a multi-agent AI system that runs institutional-grade risk math and real-time news intelligence on any portfolio in seconds — then delivers a single plain-English verdict on your true exposure. No terminal. No expertise required.

---

## What It Does

Most portfolio trackers show you what you own. MERIDIAN shows you what you're *actually* holding.

Drop in any set of tickers and positions. Within seconds you get:

- **True factor exposure** — not just how many stocks you hold, but how many independent bets they represent
- **News intelligence** — every headline for every holding graded by evidence strength, not tone
- **A unified risk narrative** — one coherent verdict that connects your math and your news together

---

## How It Works

MERIDIAN runs three collaborating agents in sequence.

### Stage 1 — Risk Math

The risk agent fetches 252 days of real price history and computes:

- **Sector concentration** vs. S&P 500 benchmarks
- **Weighted portfolio beta** — how hard a market move hits you
- **Peak-to-trough drawdowns** per holding
- **Pairwise correlation matrix** across all positions

The centrepiece is the **Factor Compression Engine**. It runs eigenvalue decomposition on the return correlation matrix to compute **Effective N** — the true number of independent risk factors in the portfolio:

$$N_{\text{eff}} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}$$

A portfolio of 16 stocks with $N_{\text{eff}} = 3.8$ isn't diversified — it's concentrated in roughly 4 hidden factors. Greedy agglomerative clustering then labels which stocks belong to each hidden cluster.

### Stage 2 — News Intelligence

The market intelligence agent pulls live news for every holding and grades each catalyst on a **−3 to +3 scale** by materiality and evidence strength alone — not tone, not recency.

| Signal | Grade |
|---|---|
| Confirmed revenue miss | −3 |
| Regulatory probe opened | −2 |
| Analyst price target (no reasoning) | 0 |
| Confirmed contract win | +2 |

Portfolio sentiment is then a position-weighted aggregate across all holdings.

### Stage 3 — Synthesis

Both signals feed **Gemini 2.5 Flash**, which produces a single coherent risk narrative that knows your specific holdings, their correlations, and which catalysts actually move your needle.

> *"Portfolio risk elevated — tech factor compression masks true single-factor exposure; TSLA earnings miss (−3) amplifies existing high-beta drawdown risk. Reduce momentum-correlated positions before next Fed window."*

---

## Stack

| Layer | Technology |
|---|---|
| Agents & orchestration | Python · LangGraph |
| Data | yFinance |
| Risk math | NumPy · Pandas (Eigenvalue PCA, greedy agglomerative clustering) |
| Reasoning | Gemini 2.5 Flash |
| Interface | Streamlit |

---

## Running Locally
```bash
git clone https://github.com/KivancSerefoglu/MERIDIAN-AI-Portfolio-Optimization-
cd MERIDIAN-AI-Portfolio-Optimization-
pip install -r requirements.txt
streamlit run app.py
```

You will need a Gemini API key. Get one free at [aistudio.google.com](https://aistudio.google.com).

---

## Built At

**Scarlet Hacks** — April 2026

**Team:** Ferit Ozdaban · Kıvanç Şerefoğlu

---

## License

MIT
