"""Named Parquet rows for assertions about prompt gaps and completion alignment."""

from fireworks.training.sdk.routing import RoutingReferences


def replay_refs(labels):
    if labels is None or isinstance(labels, RoutingReferences):
        return labels
    files, spans = [], []
    for i, label in enumerate(labels):
        span = {"input_token_start": i, "count": 1}
        if label:
            span.update(file_index=len(files), file_row_start=0)
            files.append({"artifact_id": label, "format": "parquet_v1", "row_count": 1, "store_id": "test"})
        spans.append(span)
    return RoutingReferences(len(labels), tuple(files), tuple(spans))


def replay_rows(refs):
    if refs is None:
        return None
    if isinstance(refs, dict):
        refs = RoutingReferences.from_dict(refs)
    return [
        refs.files[span["file_index"]]["artifact_id"] if "file_index" in span else ""
        for span in refs.spans for _ in range(span["count"])
    ]
