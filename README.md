# PokerCoach Backend

FastAPI backend for the PokerCoach iOS prototype.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docker:

```bash
docker build -t pokercoach-backend:dev .
docker run --rm --env-file .env.local -p 8000:8000 --name pokercoach-backend pokercoach-backend:dev
```

## LLM Agent

The backend can use an OpenAI-compatible Responses API provider. Keep secrets in `.env.local`; it is ignored by git.

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.4-mini
```

Agent boundaries:

- The Agent understands the user question, chooses which backend tools to call, then writes concise coaching text from the returned tool results.
- Backend tools provide poker facts: showdown hand class, winner, stack bucket, pot context, EV candidates, mistake reason, and recommended action.
- Current tool catalog includes `showdown_evaluator`, `answer_validator`, `stack_context`, `position_and_stack_classifier`, `pot_context`, `decision_validator`, `spot_snapshot`, and `ev_action_compare`.
- The LLM must not change tool facts, answers, hand classes, winners, EV values, or recommendations.
- If the provider is not configured or fails, the API falls back to deterministic coaching.

Tests set `POKERCOACH_LLM_ENABLED=false`, so they never call a live model.

## Test

```bash
PYTHONPATH=. python -m pytest tests
```
