from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from commissioners.common.models import MembershipSnapshot, PolicyMembershipEventChange, PolicyMembershipEventEvidence
from commissioners.common.ruleset_strategy.config import DivisionStageConfig


@dataclass
class EffectiveMemberships:
    _memberships: dict[tuple[UUID, UUID], MembershipSnapshot] = field(default_factory=dict)

    def get(self, round_id: UUID, membership: MembershipSnapshot) -> MembershipSnapshot:
        return self._memberships.get((round_id, membership.id), membership)

    def apply(
        self,
        round_id: UUID,
        memberships: list[MembershipSnapshot],
        changes: list[PolicyMembershipEventChange],
    ) -> None:
        memberships_by_id = {membership.id: membership for membership in memberships}
        for change in changes:
            membership = memberships_by_id.get(change.league_policy_membership_id)
            if membership is None:
                continue
            self._apply(round_id, membership, change)

    def _apply(
        self,
        round_id: UUID,
        membership: MembershipSnapshot,
        change: PolicyMembershipEventChange,
    ) -> None:
        effective = self.get(round_id, membership)
        self._memberships[(round_id, membership.id)] = effective.model_copy(
            update={
                "division_id": change.to_division_id or effective.division_id,
                "status": change.status,
                "substatus": change.substatus,
                "is_champion": effective.is_champion if change.status == "competing" else False,
            }
        )


class QualifierStageTransitionKind(StrEnum):
    duplicate = "duplicate"
    failed = "failed"
    progressed = "progressed"
    promoted = "promoted"


@dataclass(frozen=True)
class QualifierStageTransition:
    kind: QualifierStageTransitionKind
    change: PolicyMembershipEventChange | None = None


@dataclass
class QualifierStageProgress:
    """State machine for one membership's parallel qualifier round.

    Per (round, membership), each stage can complete successfully once.
    Successful stages move through:
        unseen -> progressed -> promoted
    A disqualifying stage emits a failed transition. The EffectiveMemberships
    overlay then makes later callbacks see the membership as disqualified, so
    they no longer match the qualifying entrant selector.
    """

    _completed_stage_ids: dict[tuple[UUID, UUID], set[str]] = field(default_factory=dict)
    _successful_stage_changes: dict[tuple[UUID, UUID], dict[str, PolicyMembershipEventChange]] = field(
        default_factory=dict
    )

    def advance(
        self,
        *,
        round_id: UUID,
        membership: MembershipSnapshot,
        stage: DivisionStageConfig,
        stages: list[DivisionStageConfig],
        change: PolicyMembershipEventChange,
    ) -> QualifierStageTransition:
        if change.status == "disqualified":
            return QualifierStageTransition(
                kind=QualifierStageTransitionKind.failed,
                change=self._failure_change(
                    round_id=round_id,
                    membership_id=membership.id,
                    stage=stage,
                    stages=stages,
                    change=change,
                ),
            )
        return self._record_success(round_id=round_id, membership=membership, stage=stage, stages=stages, change=change)

    def _failure_change(
        self,
        *,
        round_id: UUID,
        membership_id: UUID,
        stage: DivisionStageConfig,
        stages: list[DivisionStageConfig],
        change: PolicyMembershipEventChange,
    ) -> PolicyMembershipEventChange:
        return change.model_copy(
            update={
                "notes": self._stage_progress_notes(
                    stages=stages,
                    completed_stage_ids=self._completed_stage_ids.get((round_id, membership_id), set()),
                    failed_stage_id=stage.id,
                )
            }
        )

    def _record_success(
        self,
        *,
        round_id: UUID,
        membership: MembershipSnapshot,
        stage: DivisionStageConfig,
        stages: list[DivisionStageConfig],
        change: PolicyMembershipEventChange,
    ) -> QualifierStageTransition:
        key = (round_id, membership.id)
        completed_stage_ids = self._completed_stage_ids.setdefault(key, set())
        successful_changes = self._successful_stage_changes.setdefault(key, {})
        if stage.id in completed_stage_ids:
            return QualifierStageTransition(kind=QualifierStageTransitionKind.duplicate)

        completed_stage_ids.add(stage.id)
        successful_changes[stage.id] = change
        notes = self._stage_progress_notes(stages=stages, completed_stage_ids=completed_stage_ids)
        if len(completed_stage_ids) == len(stages):
            promotion_change = self._promotion_change(stages=stages, successful_changes=successful_changes) or change
            return QualifierStageTransition(
                kind=QualifierStageTransitionKind.promoted,
                change=promotion_change.model_copy(
                    update={
                        "reason": promotion_change.reason or "completed all qualifier stages",
                        "notes": notes,
                        "evidence": [
                            *promotion_change.evidence,
                            self._stage_progress_evidence(stages=stages, completed_stage_ids=completed_stage_ids),
                        ],
                    }
                ),
            )

        return QualifierStageTransition(
            kind=QualifierStageTransitionKind.progressed,
            change=PolicyMembershipEventChange(
                league_policy_membership_id=membership.id,
                from_division_id=membership.division_id,
                to_division_id=membership.division_id,
                status="qualifying",
                substatus=f"{len(completed_stage_ids)}/{len(stages)} stages completed",
                reason=f"completed qualifier stage {stage.schedule.label}",
                notes=notes,
                evidence=[self._stage_progress_evidence(stages=stages, completed_stage_ids=completed_stage_ids)],
            ),
        )

    def _promotion_change(
        self,
        *,
        stages: list[DivisionStageConfig],
        successful_changes: dict[str, PolicyMembershipEventChange],
    ) -> PolicyMembershipEventChange | None:
        for stage in reversed(stages):
            change = successful_changes.get(stage.id)
            if change is not None and (change.to_division_id is not None or change.status == "competing"):
                return change
        return next(reversed(successful_changes.values()), None) if successful_changes else None

    def _stage_progress_notes(
        self,
        *,
        stages: list[DivisionStageConfig],
        completed_stage_ids: set[str],
        failed_stage_id: str | None = None,
    ) -> str:
        labels_by_id = {stage.id: stage.schedule.label for stage in stages}
        completed = [labels_by_id[stage.id] for stage in stages if stage.id in completed_stage_ids]
        pending = [
            labels_by_id[stage.id]
            for stage in stages
            if stage.id not in completed_stage_ids and stage.id != failed_stage_id
        ]
        parts = [f"Completed stages: {', '.join(completed) if completed else 'none'}."]
        if pending:
            parts.append(f"Pending stages: {', '.join(pending)}.")
        if failed_stage_id is not None:
            parts.append(f"Failed stage: {labels_by_id.get(failed_stage_id, failed_stage_id)}.")
        return " ".join(parts)

    def _stage_progress_evidence(
        self,
        *,
        stages: list[DivisionStageConfig],
        completed_stage_ids: set[str],
    ) -> PolicyMembershipEventEvidence:
        return PolicyMembershipEventEvidence(
            type="ruleset_stage_progress",
            title="Qualifier stage progress",
            summary=f"{len(completed_stage_ids)}/{len(stages)} qualifier stages completed",
            metadata={
                "completed_stage_ids": [stage.id for stage in stages if stage.id in completed_stage_ids],
                "pending_stage_ids": [stage.id for stage in stages if stage.id not in completed_stage_ids],
            },
        )
