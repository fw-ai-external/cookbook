"""Regenerate tokenizer defaults from the training renderer identity manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

START = "# BEGIN GENERATED TOKENIZER MODELS\n"
END = "# END GENERATED TOKENIZER MODELS"

# Experiments and trained adapters are not official base-model defaults.
EXPERIMENT_PREFIXES = ("nexa-", "exa-", "tb-")


def render_catalog(manifest: dict) -> str:
    if manifest.get("version") != 1 or manifest.get("source_account") != "fireworks":
        raise ValueError("Expected a version 1 Fireworks catalog manifest")
    identities: dict[str, set[str]] = defaultdict(set)
    for entry in manifest["renderer_ids"]:
        for model in entry["fireworks_models"]:
            prefix = "accounts/fireworks/models/"
            if model.startswith(prefix) and not model.removeprefix(prefix).startswith(EXPERIMENT_PREFIXES):
                identities[model].add(entry["id"])
    if not identities:
        raise ValueError("The manifest contains no Fireworks catalog models")
    lines = ["TOKENIZER_MODELS: dict[str, str] = {"]
    for model, repos in sorted(identities.items()):
        if len(repos) == 1:
            lines.append(f"    {json.dumps(model)}: {json.dumps(next(iter(repos)))},")
    lines += ["}", "", "AMBIGUOUS_TOKENIZER_MODELS: dict[str, tuple[str, ...]] = {"]
    for model, repos in sorted(identities.items()):
        if len(repos) > 1:
            values = ", ".join(json.dumps(repo) for repo in sorted(repos))
            lines.append(f"    {json.dumps(model)}: ({values}),")
    lines += ["}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model_defaults.py",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.output.read_text()
    before, start, tail = source.partition(START)
    _, end, after = tail.partition(END)
    if not start or not end:
        raise ValueError("The output module is missing its generated catalog markers")
    catalog = render_catalog(json.loads(args.manifest.read_text()))
    updated = before + START + catalog + END + after
    if args.check:
        if updated != source:
            print("Tokenizer defaults differ from the manifest; regenerate them.")
            return 1
    else:
        args.output.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
