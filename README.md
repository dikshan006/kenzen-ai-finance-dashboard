
***

## KenZen AI Finance Dashboard – README

```markdown
# KenZen AI Finance Dashboard – Spending Analytics

KenZen Finance is a self-serve dashboard that turns raw CSV transaction data into clear spending insights, anomaly detection, and savings opportunities.

## What It Does

- Lets users upload CSVs with transaction history.
- Automatically parses and visualizes spending by category and over time.
- Flags unusual spending patterns using statistical + rule-based anomaly detection.

## Tech Stack

- Frontend: Streamlit (Python)
- Data: Pandas, Plotly
- Analytics: Custom anomaly detection logic (z-scores + rules)
- Hosting: Streamlit Cloud

## Key Features

- CSV upload with automatic parsing and validation.
- Category breakdowns, trends, and daily/weekly views.
- Anomaly detection that highlights outlier transactions and suspicious spikes.
- Dark-mode UI with interactive charts built in Plotly.

## Live Demo

- Live: https://kenzen-finance.streamlit.app/
- Repo: https://github.com/dikshan006/kenzen-ai-finance-dashboard

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd kenzen-ai-finance-dashboard
