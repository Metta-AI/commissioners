from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from commissioners.common.commissioners import BaselineCommissioner
from commissioners.common.models import (
    DivisionCommissionerDescriptionPublic,
    DivisionDescriptionContext,
    DivisionLeaderboardContext,
    DivisionLeaderboardSnapshot,
    EpisodeResult,
    MembershipSnapshot,
    OnRoundCompletedContext,
    OnRoundCompletedResult,
    PolicyMembershipEventChange,
    PolicyPool,
    PolicyPoolEntry,
    Round,
    RoundSpec,
    ScheduleContext,
    V2RoundConfig,
)
from commissioners.common.protocol import (
    EpisodeCompletedRequest,
    EpisodeCompletedResponse,
    EpisodeResult as CommissionerProtocolEpisodeResult,
    EpisodeRequest as CommissionerProtocolEpisodeRequest,
    RoundComplete as CommissionerRoundComplete,
    RoundStart as CommissionerRoundStart,
    ScheduleEpisodes as CommissionerScheduleEpisodes,
)
from commissioners.common.utils import (
    _count_text,
    _current_schedule_slot,
    _duration_text,
    _leaderboard_rules_description,
    _plural_word,
    _round_structure_description,
    _schedule_slot_description,
)
from commissioners.common.ruleset_strategy.config import (
    ChangeMatch,
    DivisionRule,
    DivisionStageConfig,
    RulesetDivisionConfig,
    RulesetStrategyCommissionerConfig,
    load_image_ruleset_strategy_config,
)
from commissioners.common.ruleset_strategy.entrants import division_entries, select_rule
from commissioners.common.ruleset_strategy.membership_events import (
    build_membership_events,
    protocol_policy_membership_event,
    transition_change,
)
from commissioners.common.ruleset_strategy.round_state import EffectiveMemberships, QualifierStageProgress
from commissioners.common.ruleset_strategy.round_start import RoundStartView
from commissioners.common.ruleset_strategy.scheduling import schedule_entries

_STAGE_REQUEST_SEPARATOR = ":"


@dataclass(frozen=True)
class ParallelQualifierStageGroup:
    division: RulesetDivisionConfig
    stages: list[DivisionStageConfig]
    rule: DivisionRule


@dataclass(frozen=True)
class CompletedParallelQualifierStage:
    stage: DivisionStageConfig
    scheduled_episode: CommissionerProtocolEpisodeRequest


class RulesetStrategyCommissioner(BaselineCommissioner):
    """Commissioner whose scheduling, seating, ranking metadata, and membership changes come from config."""

    def __init__(self, config: RulesetStrategyCommissionerConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self._ruleset_config = load_image_ruleset_strategy_config()
        elif isinstance(config, RulesetStrategyCommissionerConfig):
            self._ruleset_config = config
        else:
            self._ruleset_config = RulesetStrategyCommissionerConfig.from_mapping(config)
        self._effective_memberships = EffectiveMemberships()
        self._stage_progress = QualifierStageProgress()

    def _config(self) -> RulesetStrategyCommissionerConfig:
        return self._ruleset_config

    def rank_division(self, ctx: DivisionLeaderboardContext) -> list[DivisionLeaderboardSnapshot]:
        config = self._config()
        if config.ranking.filter_metadata:
            filtered = [
                result
                for result in ctx.round_results
                if all(result.result_metadata.get(key) == value for key, value in config.ranking.filter_metadata.items())
            ]
            ctx = ctx.model_copy(update={"round_results": filtered})
        return super().rank_division(ctx)

    def _leaderboard_ewma_halflife(self, ctx: DivisionLeaderboardContext) -> timedelta:
        config = self._config()
        return timedelta(hours=config.ranking.ewma_halflife_hours)

    def describe_division(self, ctx: DivisionDescriptionContext) -> DivisionCommissionerDescriptionPublic:
        config = self._config()
        memberships = list(ctx.active_memberships)
        rule = select_rule(config, ctx.division, memberships)
        stages = rule.stages if rule and rule.stages is not None else config.stages
        minimum_entrants = rule.minimum_entrants if rule is not None else 1
        entrants = division_entries(ctx.division, memberships, rule)
        active_round = next((r for r in ctx.recent_rounds if r.status in ("pending", "claimed", "running")), None)
        next_round = None
        if len(entrants) < minimum_entrants:
            needed = minimum_entrants - len(entrants)
            next_round = f"Add {needed} more {_plural_word(needed, 'entrant')} before scheduling can continue."
        elif active_round is not None:
            next_round = f"The next round waits for round #{active_round.round_number} to finish."
        return DivisionCommissionerDescriptionPublic(
            round_schedule=(
                f"Rounds start every {_duration_text(config.schedule_interval_minutes)}"
                f"{_schedule_slot_description(config)} if there are at least "
                f"{_count_text(minimum_entrants)} {_plural_word(minimum_entrants, 'entrant')} in the division."
            ),
            next_round=next_round,
            round_structure=_round_structure_description(stages),
            leaderboard_rules=_leaderboard_rules_description(),
            scoring_mechanics=config.scoring_mechanics,
        )

    def schedule_rounds(self, ctx: ScheduleContext) -> list[RoundSpec]:
        config = self._config()
        current_slot = _current_schedule_slot(datetime.now(UTC), config)
        specs: list[RoundSpec] = []
        for division in ctx.divisions:
            division_rounds = [r for r in ctx.recent_rounds if r.division_id == division.id]
            if any(r.status in ("pending", "claimed", "running") for r in division_rounds):
                continue
            latest_round = max(division_rounds, key=lambda r: r.created_at, default=None)
            if latest_round is not None and latest_round.created_at >= current_slot:
                continue

            rule = select_rule(config, division, ctx.active_memberships, require_minimum=True)
            if rule is None:
                continue
            entrants = division_entries(division, ctx.active_memberships, rule)
            specs.append(
                RoundSpec(
                    division_id=division.id,
                    round_config=V2RoundConfig(
                        stages=rule.stages if rule.stages is not None else config.stages,
                        entrant_policy_version_ids=[entry.policy_version_id for entry in entrants],
                    ),
                    execution_backend=config.default_execution_backend,
                    notes=f"auto-scheduled by {type(self).__name__}:{rule.id}",
                )
            )
        return specs

    def schedule_episodes_for_round_start(self, round_start: CommissionerRoundStart) -> CommissionerScheduleEpisodes:
        config = self._config()
        view = RoundStartView(round_start, config)
        stage_group = self._parallel_qualifier_stage_group(view)
        if stage_group is not None:
            variant_id, num_agents = view.variant()
            entries = view.entries(stage_group.rule)
            return self._schedule_parallel_stages(
                view=view,
                stages=stage_group.stages,
                entries=entries,
                variant_id=variant_id,
                num_agents=num_agents,
            )

        rule = select_rule(config, view.current_division, view.memberships)
        variant_id, num_agents = view.variant()
        entries = view.entries(rule)
        return schedule_entries(
            pool=view.pool(rule),
            primary_entries=entries,
            filler_entries=view.filler_entries(entries),
            num_agents=num_agents,
            variant_id=variant_id,
            config=config,
        )

    def on_episode_complete(self, request: EpisodeCompletedRequest) -> EpisodeCompletedResponse:
        config = self._config()
        view = RoundStartView(request.round_start, config)
        stage_group = self._parallel_qualifier_stage_group(view)
        if stage_group is None:
            return super().on_episode_complete(request)

        completed_stage = self._completed_parallel_qualifier_stage(
            view=view,
            stage_group=stage_group,
            request=request,
        )
        if completed_stage is None:
            return EpisodeCompletedResponse()

        event_changes = self._parallel_qualifier_stage_changes(
            view=view,
            stage_group=stage_group,
            completed_stage=completed_stage,
            request=request,
        )
        self._effective_memberships.apply(view.round_start.round_id, view.memberships, event_changes)
        return EpisodeCompletedResponse(
            policy_membership_events=[protocol_policy_membership_event(change) for change in event_changes]
        )

    def _parallel_qualifier_stage_changes(
        self,
        *,
        view: RoundStartView,
        stage_group: ParallelQualifierStageGroup,
        completed_stage: CompletedParallelQualifierStage,
        request: EpisodeCompletedRequest,
    ) -> list[PolicyMembershipEventChange]:
        config = self._config()
        score_by_policy = self._episode_score_by_policy(request.episode_result)
        transition_rule = config._transition_rule(
            match=ChangeMatch(
                division=stage_group.division.match,
                membership=config._entrant_selector(stage_group.division.entrants, stage_group.division.match),
            ),
            transitions=completed_stage.stage.on_episode_complete,
        )
        event_changes: list[PolicyMembershipEventChange] = []
        for effective_membership in self._scheduled_stage_memberships(
            view=view,
            scheduled_episode=completed_stage.scheduled_episode,
            score_by_policy=score_by_policy,
            require_score=request.episode_result is not None,
        ):
            score = score_by_policy.get(effective_membership.policy_version_id, 0.0)
            if not transition_rule.match.matches(view.current_division, effective_membership):
                continue
            change = transition_change(
                transition_rule,
                effective_membership,
                view.divisions,
                completed_episodes=1 if effective_membership.policy_version_id in score_by_policy else 0,
                score=score,
            )
            if change is None:
                continue
            transition = self._stage_progress.advance(
                round_id=view.round_start.round_id,
                membership=effective_membership,
                stage=completed_stage.stage,
                stages=stage_group.stages,
                change=change,
            )
            if transition.change is not None:
                event_changes.append(transition.change)
        return event_changes

    def _scheduled_stage_memberships(
        self,
        *,
        view: RoundStartView,
        scheduled_episode: CommissionerProtocolEpisodeRequest,
        score_by_policy: dict[UUID, float],
        require_score: bool,
    ) -> list[MembershipSnapshot]:
        memberships = []
        for membership in view.memberships:
            effective_membership = self._effective_memberships.get(view.round_start.round_id, membership)
            if effective_membership.division_id != view.current_division.id:
                continue
            if effective_membership.policy_version_id not in scheduled_episode.policy_version_ids:
                continue
            if require_score and effective_membership.policy_version_id not in score_by_policy:
                continue
            memberships.append(effective_membership)
        return memberships

    def schedule_episodes(
        self,
        *,
        pool: PolicyPool,
        entries: list[PolicyPoolEntry],
        num_agents: int,
        variant_id: str,
    ) -> CommissionerScheduleEpisodes:
        config = self._config()
        return schedule_entries(
            pool=pool,
            primary_entries=entries,
            filler_entries=[],
            num_agents=num_agents,
            variant_id=variant_id,
            config=config,
        )

    def complete_round_for_round_start(
        self,
        round_start: CommissionerRoundStart,
        episode_results: list[CommissionerProtocolEpisodeResult],
        scheduled_episodes: list[CommissionerProtocolEpisodeRequest] | None = None,
    ) -> CommissionerRoundComplete:
        config = self._config()
        view = RoundStartView(round_start, config)
        rule = select_rule(config, view.current_division, view.memberships)
        entries = view.entries(rule)
        local_episode_results = view.episode_results(episode_results)
        complete = self.complete_round(
            round_row=view.round_row(),
            pool=view.pool(rule),
            entries=entries,
            episode_results=local_episode_results,
        )
        if self._parallel_qualifier_stage_group(view) is not None:
            return complete
        hook = self.on_round_completed(
            view.on_round_completed_context(
                complete,
                episode_results=local_episode_results,
                scheduled_episodes=scheduled_episodes,
            )
        )
        complete.policy_membership_events = [
            protocol_policy_membership_event(change) for change in hook.policy_membership_events
        ]
        return complete

    def complete_round(
        self,
        *,
        round_row: Round,
        pool: PolicyPool,
        entries: list[PolicyPoolEntry],
        episode_results: list[EpisodeResult],
    ) -> CommissionerRoundComplete:
        complete = super().complete_round(
            round_row=round_row,
            pool=pool,
            entries=entries,
            episode_results=episode_results,
        )
        config = self._config()
        if config.ranking.result_metadata:
            for division_ranking in complete.results:
                for ranking in division_ranking.rankings:
                    ranking.result_metadata = dict(ranking.result_metadata) | dict(config.ranking.result_metadata)
        return complete

    def on_round_completed(self, ctx: OnRoundCompletedContext) -> OnRoundCompletedResult:
        config = self._config()
        if not config.membership_changes:
            return super().on_round_completed(ctx)
        return OnRoundCompletedResult(policy_membership_events=build_membership_events(ctx, config))

    def _parallel_qualifier_stage_group(
        self,
        view: RoundStartView,
    ) -> ParallelQualifierStageGroup | None:
        if view.current_division.type != "staging":
            return None
        config = self._config()
        for division_key, division in config.divisions.items():
            if not division.match.matches(view.current_division):
                continue
            stages = config._expanded_stages(division)
            if len(stages) <= 1:
                return None
            rule = DivisionRule(
                id=f"{division_key}-all-stages",
                match=division.match,
                entrants=config._entrant_selector(division.entrants, division.match),
                minimum_entrants=division.min_entries_to_start or config.defaults.min_entries_to_start,
                stages=[stage.schedule.to_stage_config() for stage in stages],
            )
            return ParallelQualifierStageGroup(
                division=division,
                stages=stages,
                rule=rule,
            )
        return None

    def _completed_parallel_qualifier_stage(
        self,
        *,
        view: RoundStartView,
        stage_group: ParallelQualifierStageGroup,
        request: EpisodeCompletedRequest,
    ) -> CompletedParallelQualifierStage | None:
        completed_event = request.episode_result or request.episode_failed
        assert completed_event is not None, "EpisodeCompletedRequest requires one completed event"
        stage_id = self._stage_id_from_request_id(completed_event.request_id)
        if stage_id is None:
            return None
        stage = next((candidate for candidate in stage_group.stages if candidate.id == stage_id), None)
        if stage is None or not stage.on_episode_complete:
            return None

        scheduled_episode = self._scheduled_parallel_stage_episodes(view, stage_group).get(completed_event.request_id)
        if scheduled_episode is None:
            return None
        return CompletedParallelQualifierStage(stage=stage, scheduled_episode=scheduled_episode)

    def _scheduled_parallel_stage_episodes(
        self,
        view: RoundStartView,
        stage_group: ParallelQualifierStageGroup,
    ) -> dict[str, CommissionerProtocolEpisodeRequest]:
        variant_id, num_agents = view.variant()
        schedule = self._schedule_parallel_stages(
            view=view,
            stages=stage_group.stages,
            entries=view.entries(stage_group.rule),
            variant_id=variant_id,
            num_agents=num_agents,
        )
        return {episode.request_id: episode for episode in schedule.episodes}

    def _schedule_parallel_stages(
        self,
        *,
        view: RoundStartView,
        stages: list[DivisionStageConfig],
        entries: list[PolicyPoolEntry],
        variant_id: str,
        num_agents: int,
    ) -> CommissionerScheduleEpisodes:
        episodes: list[CommissionerProtocolEpisodeRequest] = []
        for stage in stages:
            pool = PolicyPool(
                id=view.round_start.round_id,
                label=stage.schedule.label,
                pool_type="round",
                config=stage.schedule.to_stage_config().model_dump(mode="json"),
            )
            stage_schedule = schedule_entries(
                pool=pool,
                primary_entries=entries,
                filler_entries=view.filler_entries(entries),
                num_agents=num_agents,
                variant_id=variant_id,
                config=self._config(),
            )
            for episode in stage_schedule.episodes:
                episodes.append(
                    episode.model_copy(
                        update={
                            "request_id": self._stage_request_id(stage.id, episode.request_id),
                            "tags": dict(episode.tags)
                            | {
                                "ruleset_stage_id": stage.id,
                                "ruleset_stage_label": stage.schedule.label,
                            },
                        }
                    )
                )
        return CommissionerScheduleEpisodes(episodes=episodes)

    def _episode_score_by_policy(
        self,
        episode_result: CommissionerProtocolEpisodeResult | None,
    ) -> dict[UUID, float]:
        if episode_result is None:
            return {}
        scores: dict[UUID, list[float]] = {}
        for score in episode_result.scores:
            scores.setdefault(score.policy_version_id, []).append(score.score)
        return {policy_id: sum(policy_scores) / len(policy_scores) for policy_id, policy_scores in scores.items()}

    def _stage_request_id(self, stage_id: str, request_id: str) -> str:
        return f"{stage_id}{_STAGE_REQUEST_SEPARATOR}{request_id}"

    def _stage_id_from_request_id(self, request_id: str) -> str | None:
        if _STAGE_REQUEST_SEPARATOR not in request_id:
            return None
        stage_id, _rest = request_id.split(_STAGE_REQUEST_SEPARATOR, 1)
        return stage_id or None
