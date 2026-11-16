"""Studio control protocol, version 1. External records are validated here."""
from __future__ import annotations

from typing import Literal
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Birth(Record):
    boot_id: str = Field(min_length=1, max_length=200)
    service_id: str = Field(min_length=1, max_length=200)


class Identity(Record):
    instance_id: str = Field(min_length=1, max_length=200)
    birth: Birth


class Expected(Identity):
    revision: int = Field(ge=0)


class ResourceIdentity(Identity):
    gpu_uuid: str = Field(pattern=r"^GPU-[A-Za-z0-9-]{8,100}$")
    resource_id: str

    @model_validator(mode="after")
    def canonical_resource(self):
        if self.resource_id != "gpu:" + self.gpu_uuid:
            raise ValueError("resource_id must name the physical GPU UUID")
        return self


class ProcessBirth(Record):
    boot_id: str
    service_id: str
    pid: int = Field(gt=0)
    start_ticks: str = Field(min_length=1)


class WorkerSetting(Identity):
    worker: str = Field(min_length=1, max_length=200)
    resource_id: str = Field(pattern=r"^gpu:GPU-[A-Za-z0-9-]{8,100}$")
    enabled: bool


class ControlEffect(Record):
    effect_id: str = Field(min_length=1, max_length=200)
    kind: Literal["unload_comfy", "configure_workers", "shutdown"]
    expected_engine_birth: ProcessBirth | None = None
    workers: list[WorkerSetting] = Field(default_factory=list)
    sparks: int | None = Field(default=None, ge=1, le=8)

    @model_validator(mode="after")
    def scoped_effect(self):
        if self.kind == "unload_comfy" and self.expected_engine_birth is None:
            raise ValueError("unload requires the exact engine birth")
        if self.kind != "configure_workers" and (self.workers or self.sparks is not None):
            raise ValueError("only a worker settings effect changes workers")
        return self


class Scope(Record):
    resources: list[str] = Field(min_length=1, max_length=64)

    @field_validator("resources")
    @classmethod
    def stable_resources(cls, value):
        if value != sorted(set(value)) or any(not item or len(item) > 200 or any(ord(c) < 32 for c in item) for item in value):
            raise ValueError("resources must be sorted unique nonempty inventory IDs")
        return value


class FenceTransfer(Scope):
    operation_id: str = Field(min_length=1, max_length=200)
    generation: int = Field(gt=0)


class FenceMutation(Scope):
    schema_version: Literal[1] = Field(alias="schema")
    expected: Expected
    operation_id: str = Field(min_length=1, max_length=200)
    generation: int = Field(gt=0)
    idempotency_key: str = Field(min_length=1, max_length=200)
    state: Literal["draining", "sealed"] | None = None
    effect: ControlEffect | None = None
    transfer: FenceTransfer | None = None

    @model_validator(mode="after")
    def exact_transfer(self):
        if self.transfer is not None:
            if (self.state != "draining" or self.effect is not None
                    or self.operation_id == self.transfer.operation_id or self.generation <= self.transfer.generation
                    or not set(self.transfer.resources) <= set(self.resources)
                    or any(re.fullmatch(r"gpu:GPU-[A-Za-z0-9-]{8,100}", item) is None
                           for item in self.resources + self.transfer.resources)):
                raise ValueError("transfer needs distinct newer owner, canonical full scope and draining state")
        return self


class Coverage(Record):
    resource: str
    activity_class: str
    status: Literal["supported", "unsupported"]
    reason: str

    @model_validator(mode="after")
    def reason_for_unsupported(self):
        if self.status == "unsupported" and not self.reason:
            raise ValueError("unsupported coverage requires a reason")
        return self


class Activity(Scope):
    activity_id: str
    root_id: str
    parent_id: str | None
    kind: str
    state: Literal["queued", "running", "uncertain", "settled"]
    attempt_id: str | None
    job_id: str | None
    outcome: str | None
    blocker: str | None
    runtime_birth: ProcessBirth | None = None
    residency_id: str | None = None


class EffectEvidence(Scope):
    effect_id: str
    operation_id: str
    generation: int = Field(gt=0)
    kind: Literal["unload_comfy", "configure_workers", "shutdown"]
    state: Literal["MayHaveApplied", "Verified", "Rejected"]
    expected_engine_birth: ProcessBirth | None
    reason: str | None


class Attempt(Scope):
    attempt_id: str
    reservation_id: str
    coordinator: Identity
    target: Identity
    generation: int = Field(gt=0)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_id: str
    state: Literal["reserved", "consumed", "settled", "uncertain"]
    job_id: str | None
    outcome: str | None


class Fence(Scope):
    operation_id: str
    generation: int = Field(gt=0)
    state: Literal["draining", "sealed"]
    roots: list[str]
    protected_resources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def protected_scope(self):
        if self.protected_resources != sorted(set(self.protected_resources)) or not set(self.protected_resources) <= set(self.resources):
            raise ValueError("protected resources must be an exact subset of the fence")
        return self


class OwnershipRecord(Scope):
    generation: int = Field(gt=0)
    original: bool
    revision: int = Field(ge=0)
    previous_resources: list[str] | None = None
    protected_resources: list[str] = Field(default_factory=list)
    successor: FenceTransfer | None = None


class Snapshot(Expected):
    schema_version: Literal[1] = Field(alias="schema")
    self_resource: ResourceIdentity | None
    worker_bindings: list[WorkerSetting] = Field(default_factory=list)
    sparks: int | None = Field(default=None, ge=1, le=8)
    coverage: list[Coverage]
    activities: list[Activity]
    attempts: list[Attempt]
    fences: list[Fence]
    blockers: list[str]
    effects: list[EffectEvidence] = Field(default_factory=list)
    ownership_history: dict[str, list[OwnershipRecord]] = Field(default_factory=dict)


class FenceReply(Snapshot):
    operation_id: str
    generation: int = Field(gt=0)
    idempotency_key: str
    outcome: Literal["draining", "sealed", "released", "blocked"]


class TicketRequest(Scope):
    schema_version: Literal[1] = Field(alias="schema")
    action: Literal["reserve", "consume", "outcome"]
    expected: Expected
    attempt_id: str = Field(min_length=1, max_length=200)
    coordinator: Identity
    generation: int = Field(gt=0)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reservation_id: str | None = None


class TicketReply(Expected):
    schema_version: Literal[1] = Field(alias="schema")
    attempt: Attempt


def document() -> dict:
    records = (FenceMutation, Snapshot, FenceReply, TicketRequest, TicketReply)
    return {"protocol": "studio-control-v1", "schemas": {record.__name__: record.model_json_schema() for record in records}}
