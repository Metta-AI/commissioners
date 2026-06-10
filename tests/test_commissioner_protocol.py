from uuid import uuid4

import pytest
from pydantic import ValidationError

from commissioners.common.protocol import CommissionerMessage, PolicyMembershipEventChange


def test_policy_membership_event_rejects_is_champion_update() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PolicyMembershipEventChange(
            league_policy_membership_id=uuid4(),
            status="competing",
            substatus="champion",
            reason="promoted",
            is_champion=True,
        )


def test_round_complete_rejects_is_champion_membership_event() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CommissionerMessage.from_json(
            {
                "type": "round_complete",
                "policy_membership_events": [
                    {
                        "league_policy_membership_id": str(uuid4()),
                        "status": "competing",
                        "substatus": "champion",
                        "reason": "promoted",
                        "is_champion": True,
                    }
                ],
            }
        )
