# SFT: Teaching a model to read receipts

> **This notebook no longer runs.** It uploads an account dataset and submits a managed SFT job, and Fireworks no longer accepts either. For new runs, use the SFT recipe (`training/examples/sft/train_sft_serverless.py`) on the Training API; see the [migration guide](https://docs.fireworks.ai/fine-tuning/training-api/migrate-from-managed).

In this example we'll take a general vision model and teach it to turn a **photo of a receipt** into clean, structured JSON — every time, in exactly the format we want.

**Is this you?** You have a job with one "right" shape of output (JSON, tags, codes) and a pile of examples showing input → correct output, but the base model only gets it right *sometimes*. If you're doing invoice parsing, form extraction, screenshot-to-JSON, or OCR-into-structured-data, you're in the right place.

**The data.** We'll use [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2): real receipt photos, each paired with the correct JSON parse. This one is **visual** — the images ride along inside the training file as base64. The notebook builds it inline (downloads CORD-v2, encodes images, and carves off a `test`-split holdout the model never trains on, so the "after" score is honest).

**The model.** We'll fine-tune `qwen3p5-9b` — small enough to be cheap to train and serve, but a vision model, so it can actually *see* the receipt. Training support is per model, so check the [Models catalog](https://docs.fireworks.ai/fine-tuning/models) before swapping in another vision base.

**The technique.** This is plain **supervised fine-tuning (SFT)**: when you can show the model gold input→output pairs, SFT is the most direct way to lock in the behavior you want.

**What we'll do.** Train in `cord_receipt_sft_sdk.ipynb` (pure Fireworks Python SDK) — the same notebook scores the base model against our tuned one (JSON F1) on receipts it has never seen, before vs. after. Heads-up: the "GO LIVE" cells spend real GPU time.
