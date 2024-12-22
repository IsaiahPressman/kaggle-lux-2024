import json
from typing import Any


class Agent:
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, Any],
    ) -> None:
        self.team_id = self.get_team_id(player)
        with open("temp.json", "w") as f:
            json.dump(env_cfg, f)

        raise NotImplementedError

    @property
    def opp_team_id(self) -> int:
        return 1 - self.team_id

    @staticmethod
    def get_team_id(player: str) -> int:
        if player == "player_0":
            return 0

        if player == "player_1":
            return 1

        raise ValueError(f"Invalid player '{player}'")
