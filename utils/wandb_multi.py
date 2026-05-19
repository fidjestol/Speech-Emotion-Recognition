from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable


SECONDARY_API_KEY_ENV = "WANDB_SECONDARY_API_KEY"
SECONDARY_PROJECT_ENV = "WANDB_SECONDARY_PROJECT"
SECONDARY_ENTITY_ENV = "WANDB_SECONDARY_ENTITY"


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class MultiRunSummaryProxy:
    def __init__(self, runs: list[Any]) -> None:
        self._runs = runs

    def __setitem__(self, key: str, value: Any) -> None:
        for run in self._runs:
            run.summary[key] = value


def _is_wandb_table(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "data") and value.__class__.__name__ == "Table"


def _clone_wandb_table(table: Any) -> Any:
    import wandb

    return wandb.Table(columns=list(table.columns), data=[list(row) for row in table.data])


def _clone_log_payload(value: Any) -> Any:
    if _is_wandb_table(value):
        return _clone_wandb_table(value)
    if isinstance(value, dict):
        return {key: _clone_log_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_log_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_log_payload(item) for item in value)
    return value


@dataclass
class MultiWandbRun:
    primary: Any
    mirror: Any | None = None

    @property
    def runs(self) -> list[Any]:
        return [run for run in (self.primary, self.mirror) if run is not None]

    @property
    def summary(self) -> MultiRunSummaryProxy:
        return MultiRunSummaryProxy(self.runs)

    @property
    def id(self) -> str:
        return str(getattr(self.primary, "id", ""))

    @property
    def name(self) -> str:
        return str(getattr(self.primary, "name", ""))

    @property
    def url(self) -> str | None:
        return getattr(self.primary, "url", None)

    @property
    def project(self) -> str | None:
        return getattr(self.primary, "project", None)

    @property
    def entity(self) -> str | None:
        return getattr(self.primary, "entity", None)

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        for run in self.runs:
            payload = _clone_log_payload(data)
            if step is None:
                run.log(payload)
            else:
                run.log(payload, step=step)

    def finish(self) -> None:
        for run in self.runs:
            run.finish()

    def log_artifact(self, artifact_factory: Callable[[], Any]) -> None:
        for run in self.runs:
            run.log_artifact(artifact_factory())


def init_multi_wandb_run(**init_kwargs: Any) -> MultiWandbRun:
    import wandb

    init_kwargs = dict(init_kwargs)
    if init_kwargs.get("reinit") is True:
        init_kwargs["reinit"] = "create_new"

    primary = wandb.init(**init_kwargs)
    secondary_api_key = os.environ.get(SECONDARY_API_KEY_ENV, "").strip()
    if not secondary_api_key:
        return MultiWandbRun(primary=primary, mirror=None)

    mirror_kwargs = dict(init_kwargs)
    mirror_kwargs["reinit"] = "create_new"
    mirror_project = os.environ.get(SECONDARY_PROJECT_ENV, "").strip()
    mirror_entity = os.environ.get(SECONDARY_ENTITY_ENV, "").strip()
    if mirror_project:
        mirror_kwargs["project"] = mirror_project
    if mirror_entity:
        mirror_kwargs["entity"] = mirror_entity

    with temporary_env({"WANDB_API_KEY": secondary_api_key}):
        mirror = wandb.init(**mirror_kwargs)
    return MultiWandbRun(primary=primary, mirror=mirror)
