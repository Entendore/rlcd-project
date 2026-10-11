.PHONY: venv install test demo clean
venv:
    python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
install:
    pip install -r requirements.txt
test:
    python -m pytest tests/ -v
demo:
    python examples/support_ticket.py
    python examples/train_demo.py
clean:
    rm -rf .venv build dist *.egg-info .pytest_cache
    find . -name __pycache__ -type d -exec rm -rf {} +
