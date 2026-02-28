AstraQuant is a full-stack quantitative trading research platform built with a production-oriented architecture.

It combines:

⚡ Real-time WebSocket market streaming

📊 Strategy backtesting engine

🔁 Multi-timeframe resampling

🧠 AI-ready modular architecture

Designed for low-latency trading research and future ML-driven strategy deployment.

🎯 Project Motivation

Modern retail traders lack access to professional-grade infrastructure for:

Real-time market streaming

Strategy research

Performance testing

Scalable backtesting

AstraQuant bridges that gap using modern backend systems and clean frontend architecture.

🏗 System Architecture
React / Next.js Frontend
        │
        │ WebSocket + REST
        ▼
FastAPI Backend
        │
        ├── Market Data Stream
        ├── Strategy Engine
        ├── Timeframe Resampler
        └── Backtesting Core
Why This Architecture?

FastAPI → High-performance async backend

WebSockets → Real-time low-latency updates

Modular Strategy Engine → Easy extension

Clean separation of concerns

This design allows seamless scaling into AI-driven trading systems.

🔥 Core Features
1️⃣ Real-Time Market Streaming

WebSocket connection to backend

Live BTCUSDT price updates

Frontend candle update system

Instrumented debug logs:

WS CONNECTED

WS MESSAGE

Updating candle

Automatic reconnection handling

2️⃣ Strategy Backtesting Engine

Currently implemented:

Moving Average Crossover Strategy

Adjustable short & long windows

Trading cost modeling

Multi-timeframe support (1m, 5m)

Automatic Pandas resampling

Example API:

GET /backtest/ma?short=50&long=200&cost=0.001&timeframe=5m
3️⃣ Timeframe Resampling Logic

Raw 1m candle data

Resampled to higher timeframes

Backtester logic remains unchanged

Ensures consistency & modularity

This enables strategy evaluation across time granularities without code duplication.

🧠 Engineering Highlights

Async WebSocket architecture

Stateless REST endpoints

Modular strategy injection pattern

Separation of streaming vs backtesting logic

Clean API parameterization

Timeframe abstraction layer

This project demonstrates:

Backend system design

Real-time data handling

Financial algorithm implementation

Clean frontend-backend integration

Scalability planning

📂 Project Structure
astraquant/
│
├── backend/
│   ├── main.py
│   ├── backtester.py
│   ├── strategies/
│   └── data/
│
├── frontend/
│   ├── hooks.ts
│   ├── PriceChart.tsx
│   └── components/
│
└── README.md
⚙️ Local Setup
Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Runs on:

http://127.0.0.1:8000
Frontend
cd frontend
npm install
npm run dev

Runs on:

http://localhost:3000
📈 Current Strategy Logic

Moving Average Crossover

Long signal → short MA crosses above long MA

Exit signal → short MA crosses below long MA

Trading cost applied per trade

Evaluated across configurable timeframe

🚀 Roadmap
Phase 1 (Completed)

Real-time streaming

MA backtesting

Multi-timeframe support

Live chart integration

Phase 2

RSI Strategy

MACD Strategy

Multi-symbol support

Performance metrics dashboard

Portfolio backtesting

Phase 3 (AI Integration)

ML-based signal prediction

LSTM forecasting

Reinforcement Learning agent

Hyperparameter optimization

Auto strategy tuning

🛠 Future Production Enhancements

Dockerized deployment

Redis pub/sub streaming layer

PostgreSQL trade persistence

Cloud deployment (AWS/GCP)

Role-based authentication

Strategy marketplace

📌 Tech Stack

Backend:

Python

FastAPI

Pandas

Uvicorn

WebSockets

Frontend:

React

Next.js

TypeScript

Realtime chart integration

🎓 What This Project Demonstrates

Real-time systems engineering

Financial modeling implementation

Quantitative strategy development

Clean scalable architecture

AI-ready infrastructure design

⚠️ Disclaimer

This system is built for research and educational purposes.
Cryptocurrency trading involves substantial risk.

👨‍💻 Authors

Developed as a collaborative quant engineering project.
Designed with scalability and AI expansion in mind.
