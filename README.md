# RLCD — Reinforcement Learning for Calibrated Decisions

Full System One model implementation: typed primitives (Choice/Score/Noul),
non-autoregressive parallel inference, adaptive rewards, contrastive +
adversarial + curriculum training, federated calibration, uncertainty
decomposition, drift monitoring, hybrid inference, domain adapters.

## Quick start
    make venv && make test && make demo

## Layout
    config/   default_config.yaml
    rlcd/     core · calibration · training · inference · domains · evaluation · utils
    examples/ support_ticket.py · train_demo.py
    tests/
