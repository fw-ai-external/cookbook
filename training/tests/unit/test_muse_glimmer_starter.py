import json
import shlex
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from training.examples.sft import muse_glimmer_starter as starter


def test_promotion_deadline_is_end_of_nov_30_pacific():
    assert starter.promotion_active(datetime(2026, 12, 1, 7, 59, tzinfo=timezone.utc))
    assert not starter.promotion_active(datetime(2026, 12, 1, 8, 0, tzinfo=timezone.utc))


def test_serverless_names_include_a_promoted_model():
    names = starter._names("acme", "router-v1")
    assert names["serverless_model"] == "accounts/acme/models/muse-starter-sl-router-v1"
    assert starter.parse_args(["launch", "--surface", "serverless", "--no-promote"]).no_promote


def test_muse_configuration_matches_skill():
    assert starter.BASE_MODEL == "accounts/fireworks/models/muse-glimmer-30b"
    assert starter.TOKENIZER_MODEL == "meta-models/Muse-Glimmer-30B"
    assert starter.RENDERER_NAME == "muse_glimmer"
    assert starter.DEFAULT_DATASET == ""


def test_defaults_are_a_small_first_run():
    args = starter.parse_args(["launch"])
    assert args.surface == "managed"
    assert args.epochs == 1
    assert args.lora_rank == 8


def test_managed_launch_requires_a_user_dataset():
    with pytest.raises(SystemExit, match="Pass --dataset"):
        starter.cmd_launch_managed(starter.parse_args(["launch"]))


def test_serverless_plan_prints_every_resolved_training_default(monkeypatch, capsys):
    monkeypatch.setattr(starter, "_connect", lambda: (SimpleNamespace(), "acme"))
    args = starter.parse_args(["plan", "--surface", "serverless"])
    assert starter.cmd_plan(args) == 0
    params = json.loads(capsys.readouterr().out)["parameters"]
    assert params["train_on"] == "last_assistant_turn"
    assert params["loss"] == "cross_entropy"
    assert params["optimizer"] == {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    assert params["emit_grad_norm_metrics"] is True
    assert params["lora_alpha"] == starter.LORA_ALPHA
    assert params["lora_targets"] == starter.LORA_TARGETS


def test_managed_launch_refuses_to_reuse_a_job_with_different_settings(monkeypatch, capsys):
    args = starter.parse_args(["launch", "--run-id", "r1", "--lora-rank", "16", "--dataset", "user-data"])
    existing = SimpleNamespace(
        name="accounts/acme/supervisedFineTuningJobs/muse-starter-r1", state="JOB_STATE_COMPLETED",
        base_model=starter.BASE_MODEL, dataset=f"accounts/acme/datasets/{args.dataset}",
        output_model="accounts/acme/models/muse-starter-r1", epochs=args.epochs, lora_rank=8,
        learning_rate=args.learning_rate, max_context_length=args.max_context_length,
    )

    def create(**kwargs):
        raise AssertionError("must not create when the run ID exists")

    jobs = SimpleNamespace(get=lambda job_id: existing, create=create)
    monkeypatch.setattr(starter, "_connect", lambda: (SimpleNamespace(supervised_fine_tuning_jobs=jobs), "acme"))
    assert starter.cmd_launch_managed(args) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["reused"] is False
    assert out["mismatched"]["lora_rank"] == {"existing": 8, "requested": 16}


def test_serverless_prepare_splits_the_users_file(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": json.dumps({"route": "small model"})}]}
            for i in range(10)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    args = starter.parse_args(["prepare", "--run-id", "r", "--file", str(data), "--n-eval", "3",
                               "--log-path", str(tmp_path / "run")])
    run_dir, train_ex, eval_ex, _, _, report = starter._prepare(args)
    assert (len(train_ex), len(eval_ex)) == (7, 3)
    assert report["source"] == str(data)
    holdout = {json.loads(line)["messages"][0]["content"] for line in (run_dir / "holdout.jsonl").read_text().splitlines()}
    train = {json.loads(line)["messages"][0]["content"] for line in (run_dir / "train.jsonl").read_text().splitlines()}
    assert not holdout & train
    assert len(starter._messages(rows[0], False)) == 1


def test_serverless_prepare_requires_a_file():
    with pytest.raises(SystemExit):
        starter._prepare(starter.parse_args(["prepare", "--run-id", "r"]))


def test_score_uses_json_keys_when_the_answer_is_json():
    ok, fields = starter._score('Sure: {"route": "big model", "math": true}', '{"route": "big model", "math": true}')
    assert ok and fields == {"route": True, "math": True}
    ok, fields = starter._score('{"route": "small model"}', '{"route": "big model", "math": true}')
    assert not ok and fields == {"route": False, "math": False}
    assert starter._score("  Paris ", "Paris") == (True, {})
    ok, fields = starter._score('{"route": "big model"}', '{"route": "big model", "note": null}')
    assert not ok and fields == {"route": True, "note": False}


def test_watch_does_not_count_queue_time_as_a_stall(monkeypatch):
    states = iter(["JOB_STATE_PENDING"] * 3 + ["JOB_STATE_COMPLETED"])
    clock = iter(range(0, 100_000, 3_600))

    def get(job_id):
        return SimpleNamespace(state=next(states), job_progress=SimpleNamespace(percent=0, epoch=0),
                               status=None, name="j", output_model="m", evaluation_dataset="",
                               metrics_file_signed_url=None, model_dump=lambda: {})

    monkeypatch.setattr(starter, "_connect",
                        lambda: (SimpleNamespace(supervised_fine_tuning_jobs=SimpleNamespace(get=get)), "acme"))
    monkeypatch.setattr(starter.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(starter.time, "sleep", lambda s: None)
    args = starter.parse_args(["watch", "--run-id", "r", "--no-progress-minutes", "30"])
    assert starter.cmd_watch(args) == 0


def test_parse_json_tolerates_prose():
    assert starter._parse_json('Sure: {"route": "big model"} done') == {"route": "big model"}
    assert starter._parse_json("no json") == {}


def test_missing_key_never_asks_for_chat(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        starter._connect()
    message = str(exc.value)
    assert "https://app.fireworks.ai/settings/users/api-keys" in message
    assert "Never paste it into chat" in message


def test_names_are_account_relative():
    names = starter._names("acme", "router-v1")
    assert names["job"] == "accounts/acme/supervisedFineTuningJobs/muse-starter-router-v1"
    assert names["output_model"] == "accounts/acme/models/muse-starter-router-v1"


def test_upload_validation_reports_bad_rows_without_dropping(tmp_path):
    f = tmp_path / "t.jsonl"
    good = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{}"}]}
    bad = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]}
    not_list = {"messages": "hi"}
    not_dict = {"messages": [{"role": "user", "content": "hi"}, "oops"]}
    f.write_text("\n".join(json.dumps(r) for r in (good, bad, good, not_list, not_dict)) + "\n")
    rows, problems = starter._validate_chat_jsonl(f)
    assert rows == 5
    assert problems == [
        "line 2: last message is not a non-empty assistant reply",
        "line 4: messages is not a non-empty list of objects",
        "line 5: messages is not a non-empty list of objects",
    ]


def test_render_summary_covers_every_row_and_reports_fetch_errors(monkeypatch):
    rows = [
        {"renderer": "muse_glimmer", "train_on_what": "LAST_ASSISTANT_MESSAGE", "token_ids": [1] * 50,
         "training_loss_weights": [0] * 40 + [1] * 10, "image_token_count": 600},
        {"renderer": "muse_glimmer", "train_on_what": "LAST_ASSISTANT_MESSAGE", "token_ids": [1] * 90,
         "training_loss_weights": [0] * 90, "image_token_count": 972},
    ]
    monkeypatch.setattr(starter, "_fetch_lines", lambda url: [json.dumps(r) for r in rows])
    summary = starter._render_summary("signed")
    assert summary["tokens_min_max"] == [50, 90]
    assert summary["rows_with_no_trained_tokens"] == 1
    assert summary["image_tokens_min_max"] == [600, 972]

    def not_ready(url):
        raise RuntimeError("404")

    monkeypatch.setattr(starter, "_fetch_lines", not_ready)
    assert "error" in starter._render_summary("signed")


def test_cleanup_reports_real_failures(monkeypatch, capsys):
    import fireworks
    import httpx

    def resp(code):
        return httpx.Response(code, request=httpx.Request("DELETE", "https://api.fireworks.ai/x"))

    def gone(*a, **k):
        raise fireworks.NotFoundError("not found", response=resp(404), body=None)

    def denied(*a, **k):
        raise fireworks.PermissionDeniedError("denied", response=resp(403), body=None)

    client = SimpleNamespace(models=SimpleNamespace(delete=gone),
                             supervised_fine_tuning_jobs=SimpleNamespace(delete=denied))
    monkeypatch.setattr(starter, "_connect", lambda: (client, "acme"))
    assert starter.cmd_cleanup(starter.parse_args(["cleanup", "--run-id", "r"])) == 1
    out = capsys.readouterr()
    assert "does not exist" in out.out
    assert "FAILED to delete job" in out.err


RUN_REF = "acme/run-" + "0" * 32
NEXT_RUN_REF = "acme/run-" + "1" * 32


def _done(value=None):
    return SimpleNamespace(result=lambda: value)


class _FakeTrainer:
    """A serverless LoRA training client that records what the starter asks of it."""

    def __init__(self, run_ref=RUN_REF, fail_on_call=None, fail_load=False):
        self.run_ref = run_ref
        self.fail_on_call = fail_on_call
        self.fail_load = fail_load
        self.batches = []
        self.events = []

    def forward_backward(self, batch, loss_fn):
        if len(self.batches) + 1 == self.fail_on_call:
            raise RuntimeError("trainer went away")
        self.batches.append(list(batch))
        self.events.append("train")
        return _done(SimpleNamespace(metrics={"loss:sum": 2.0, "response_tokens": 4.0}))

    def optim_step(self, adam, emit_grad_norm_metrics):
        return _done(SimpleNamespace(metrics={"grad_norm": 1.0}))

    def save_state(self, name):
        self.events.append(f"save_state:{name}")
        return _done(SimpleNamespace(path=f"{self.run_ref}/{name}"))

    def save_weights_for_sampler(self, name):
        self.events.append(f"sampler:{name}")
        return _done(SimpleNamespace(path=f"{self.run_ref}/{name}"))

    def load_state_with_optimizer(self, path):
        self.events.append(f"load:{path}")
        if self.fail_load:
            raise RuntimeError("session expired")
        return _done()


def _launch_args(*extra):
    return starter.parse_args(["launch", "--surface", "serverless", "--run-id", "r1", "--file", "data.jsonl",
                               "--batch-size", "2", "--dcp-save-interval", "2", *extra])


def test_serverless_checkpoint_interval_defaults_to_20(monkeypatch, capsys):
    args = starter.parse_args(["launch", "--surface", "serverless"])
    assert (args.dcp_save_interval, args.init_from_checkpoint) == (20, None)
    monkeypatch.setattr(starter, "_connect", lambda: (SimpleNamespace(), "acme"))
    assert starter.cmd_plan(starter.parse_args(["plan", "--surface", "serverless"])) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["dcp_save_interval"] == 20
    with pytest.raises(SystemExit, match="dcp-save-interval"):
        starter.cmd_launch_serverless(_launch_args("--dcp-save-interval", "-1", "--allow-paid"))


def test_training_saves_and_logs_a_checkpoint_every_interval(tmp_path):
    trainer, progress = _FakeTrainer(), starter._Progress()
    with (tmp_path / "metrics.jsonl").open("w") as log:
        assert starter._train(trainer, list(range(10)), _launch_args(), log, progress) == 5
    assert [e for e in trainer.events if e != "train"] == ["save_state:starter-step-2", "save_state:starter-step-4"]
    assert (progress.step, progress.checkpoint, progress.checkpoint_step) == (5, f"{RUN_REF}/starter-step-4", 4)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert {"step": 4, "checkpoint": f"{RUN_REF}/starter-step-4"} in rows


def test_a_resumed_run_replays_the_batch_order_and_skips_finished_steps(tmp_path):
    full, resumed = _FakeTrainer(), _FakeTrainer()
    with (tmp_path / "metrics.jsonl").open("w") as log:
        starter._train(full, list(range(10)), _launch_args(), log, starter._Progress())
        progress = starter._Progress(2, f"{RUN_REF}/starter-step-2")
        assert starter._train(resumed, list(range(10)), _launch_args(), log, progress) == 5
    assert resumed.batches == full.batches[2:]
    assert [e for e in resumed.events if e != "train"] == ["save_state:starter-step-4"]


def test_checkpoint_step_accepts_only_the_reference_a_run_prints():
    assert starter._checkpoint_step(f"{RUN_REF}/starter-step-20") == 20
    assert starter._checkpoint_step(f" {RUN_REF}/starter-step-7-a1b2 ") == 7
    for bad in (
        "starter-step-20",
        f"{RUN_REF}/step-20",
        f"{RUN_REF}/starter-step-20x",
        "acme/run-1/starter-step-20",
        "acme/foo/starter-step-20",
        f"a/{RUN_REF}/starter-step-20",
        "Acme/run-" + "0" * 32 + "/starter-step-20",
    ):
        with pytest.raises(SystemExit, match="--init-from-checkpoint"):
            starter._checkpoint_step(bad)


def test_the_resume_command_reproduces_every_setting():
    # Every flag differs from its default, so a flag the command drops fails the comparison.
    args = _launch_args("--learning-rate", "2e-05", "--n-train", "8", "--n-eval", "7", "--seed", "3", "--epochs", "2",
                        "--lora-rank", "16", "--max-seq-len", "4096", "--max-eval-tokens", "512",
                        "--log-path", "runs/my run", "--allow-paid", "--no-promote")
    progress = starter._Progress(2, f"{RUN_REF}/starter-step-2")
    progress.step = 3
    message = starter._stopped_message(args, progress)
    assert message.startswith("The run stopped after step 3. To continue from its step-2 checkpoint")
    argv = shlex.split(message.splitlines()[-1])
    assert argv[:4] == ["python", "-m", "examples.sft.muse_glimmer_starter", "launch"]
    resumed = starter.parse_args(argv[3:])
    assert resumed.init_from_checkpoint == f"{RUN_REF}/starter-step-2"
    assert {k: v for k, v in vars(resumed).items() if k != "init_from_checkpoint"} == {
        k: v for k, v in vars(args).items() if k != "init_from_checkpoint"
    }
    assert "without a checkpoint" in starter._stopped_message(args, starter._Progress(1))


def test_a_resumed_run_saves_its_starting_point_in_its_own_session(tmp_path, capsys):
    trainer = _FakeTrainer(NEXT_RUN_REF)
    progress = starter._Progress(2, f"{RUN_REF}/starter-step-2")
    with (tmp_path / "metrics.jsonl").open("w") as log:
        starter._resume(trainer, f"{RUN_REF}/starter-step-2", progress, log)
    assert trainer.events == [f"load:{RUN_REF}/starter-step-2", "save_state:starter-step-2"]
    assert (progress.step, progress.checkpoint, progress.checkpoint_step) == (2, f"{NEXT_RUN_REF}/starter-step-2", 2)
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == {
        "resumed_from": f"{RUN_REF}/starter-step-2",
        "step": 2,
        "checkpoint": f"{NEXT_RUN_REF}/starter-step-2",
    }

    with (tmp_path / "metrics.jsonl").open("w") as log, pytest.raises(RuntimeError, match="session expired"):
        starter._resume(_FakeTrainer(fail_load=True), f"{RUN_REF}/starter-step-2", starter._Progress(2, "x"), log)
    assert "launch again without --init-from-checkpoint" in capsys.readouterr().err


class _FakeRenderer:
    def build_supervised_example(self, messages, train_on_what):
        return SimpleNamespace(to_ints=lambda: [1, 2, 3]), SimpleNamespace(tolist=lambda: [0.0, 1.0, 1.0])

    def build_generation_prompt(self, messages):
        return SimpleNamespace(to_ints=lambda: [1, 2])

    def get_stop_sequences(self):
        return []

    def parse_response(self, tokens):
        return {"role": "assistant", "content": "ok"}, SimpleNamespace(is_clean=True)


class _FakeService:
    training_session_id = "session-1"

    def __init__(self, trainer):
        self.trainer = trainer

    def create_lora_training_client(self, **kwargs):
        return self.trainer

    def create_sampling_client(self, model_path, tokenizer):
        reply = SimpleNamespace(sequences=[SimpleNamespace(tokens=[7])])
        return SimpleNamespace(sample=lambda **kwargs: _done(reply))

    def close(self):
        pass


def test_a_stopped_run_prints_a_command_that_resumes_it(tmp_path, monkeypatch, capsys):
    import fireworks.training.sdk as sdk
    import training.utils.serverless as serverless

    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "ok"}]}
            for i in range(12)]
    monkeypatch.setattr(starter, "_prepare", lambda args: (tmp_path, rows[2:], rows[:2], None, _FakeRenderer(), {}))
    monkeypatch.setattr(starter, "_connect", lambda: (None, "acme"))
    monkeypatch.setattr(serverless, "enable_serverless_supervised_409_retry", lambda service: None)
    monkeypatch.setenv("FIREWORKS_API_KEY", "unused")
    third_run = "acme/run-" + "2" * 32
    runs = [_FakeTrainer(fail_on_call=3), _FakeTrainer(NEXT_RUN_REF, fail_on_call=1), _FakeTrainer(third_run)]
    pending = list(runs)
    monkeypatch.setattr(sdk, "FiretitanServiceClient", lambda **kwargs: _FakeService(pending.pop(0)))

    def stopped_command(args):
        with pytest.raises(RuntimeError, match="trainer went away"):
            starter.cmd_launch_serverless(args)
        return capsys.readouterr().err.strip().splitlines()[-1]

    # The first run stops in step 3 and points at its step-2 checkpoint.
    first = stopped_command(_launch_args("--no-promote", "--allow-paid"))
    assert f"--init-from-checkpoint {RUN_REF}/starter-step-2" in first
    # The resumed run saves that state again in its own session, so when it stops it points there.
    second = stopped_command(starter.parse_args(shlex.split(first)[3:]))
    assert f"--init-from-checkpoint {NEXT_RUN_REF}/starter-step-2" in second
    assert runs[1].events == ["sampler:starter-base", f"load:{RUN_REF}/starter-step-2", "save_state:starter-step-2"]

    assert starter.cmd_launch_serverless(starter.parse_args(shlex.split(second)[3:])) == 0
    # The base snapshot comes from the fresh adapter, before the checkpoint loads.
    assert runs[2].events == [
        "sampler:starter-base",
        f"load:{NEXT_RUN_REF}/starter-step-2",
        "save_state:starter-step-2",
        "train",
        "train",
        "save_state:starter-step-4",
        "train",
        "sampler:starter-tuned",
    ]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (summary["steps"], summary["resumed_from"]) == (5, f"{NEXT_RUN_REF}/starter-step-2")
    assert "this attempt only" in summary["cost"]["evaluation_sampling"]["note"]

    with pytest.raises(SystemExit, match="past this run's 5 steps"):
        starter.cmd_launch_serverless(_launch_args("--init-from-checkpoint", f"{RUN_REF}/starter-step-6", "--allow-paid"))
