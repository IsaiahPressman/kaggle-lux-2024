import copy
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from lux.kit import from_json
from lux.utils import direction_to

PYTHON_SOURCE_ROOT: Final[Path] = Path(__file__).parents[1].absolute()
assert PYTHON_SOURCE_ROOT.name == "python", PYTHON_SOURCE_ROOT


class Agent:
    def __init__(self, player: str, env_cfg: dict[str, Any]) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.relic_node_positions: list[npt.NDArray[np.int_]] = []
        self.discovered_relic_nodes_ids: set[int] = set()
        self.unit_explore_locations: dict[int, tuple[int, int]] = {}
        self.all_raw_observations: list[dict[str, Any]] = []

    @property
    def opp_id(self) -> int:
        return 1 - self.team_id

    @property
    def unit_sap_range(self) -> int:
        return self.env_cfg["unit_sap_range"]

    @property
    def final_observation_step(self) -> int:
        return (self.env_cfg["max_steps_in_match"] + 1) * self.env_cfg[
            "match_count_per_episode"
        ]

    def act(
        self, step: int, obs: dict[str, Any], _remaining_overage_time: int = 60
    ) -> npt.NDArray[np.int_]:
        """implement this function to decide what actions to send to each
        available unit.

        step is the current timestep number of the game starting from 0
        going up to max_steps_in_match * match_count_per_episode - 1.
        """
        self.all_raw_observations.append(copy.deepcopy(obs))
        obs = from_json(obs)
        if len(self.all_raw_observations) == self.final_observation_step:
            self.dump_raw_observations()

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        opp_unit_mask = np.array(obs["units_mask"][self.opp_id])  # shape (max_units, )
        unit_positions = np.array(
            obs["units"]["position"][self.team_id]
        )  # shape (max_units, 2)
        opp_unit_positions = np.array(
            obs["units"]["position"][self.opp_id]
        )  # shape (max_units, 2)
        _unit_energies = np.array(
            obs["units"]["energy"][self.team_id]
        )  # shape (max_units, 1)
        observed_relic_node_positions = np.array(
            obs["relic_nodes"]
        )  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(
            obs["relic_nodes_mask"]
        )  # shape (max_relic_nodes, )
        _team_points = np.array(
            obs["team_points"]
        )  # points of each team, team_points[self.team_id] is your team's points
        team_wins = np.array(obs["team_wins"])

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # ids of opposing units you can see at this timestep
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # basic strategy here is simply to have some units randomly explore and
        # some units collecting as much energy as possible and once a relic node
        # is found, we send all units to move randomly around the first relic
        # node to gain points and information about where relic nodes are found
        # are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for relic_id in visible_relic_node_ids:
            if relic_id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(int(relic_id))
                self.relic_node_positions.append(
                    observed_relic_node_positions[relic_id]
                )

        # Stop playing if ahead
        if team_wins[self.team_id] > team_wins[self.opp_id]:
            return actions

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            if unit_id % 2 == 0:
                action = self.get_normal_action(unit_id, unit_positions[unit_id], step)
            else:
                action = self.get_combat_action(
                    unit_positions[unit_id], opp_unit_positions[opp_unit_mask]
                )

            actions[unit_id] = action

        return actions

    def get_normal_action(
        self, unit_id: int, unit_pos: npt.NDArray[np.int_], step: int
    ) -> list[int]:
        if len(self.relic_node_positions) > 0:
            nearest_relic_node_position = self.relic_node_positions[0]
            manhattan_distance = abs(
                unit_pos[0] - nearest_relic_node_position[0]
            ) + abs(unit_pos[1] - nearest_relic_node_position[1])

            # if close to the relic node we want to hover around it and hope to
            # gain points
            if manhattan_distance <= 4:
                random_direction = np.random.randint(0, 5)
                return [random_direction, 0, 0]

            # otherwise we want to move towards the relic node
            return [
                direction_to(unit_pos, nearest_relic_node_position),
                0,
                0,
            ]

        # randomly explore by picking a random location on the map and moving
        # there for about 20 steps
        if step % 20 == 0 or unit_id not in self.unit_explore_locations:
            rand_loc = (
                np.random.randint(0, self.env_cfg["map_width"]),
                np.random.randint(0, self.env_cfg["map_height"]),
            )
            self.unit_explore_locations[unit_id] = rand_loc

        return [
            direction_to(unit_pos, self.unit_explore_locations[unit_id]),
            0,
            0,
        ]

    def get_combat_action(
        self,
        unit_pos: npt.NDArray[np.int_],
        opp_unit_positions: npt.NDArray[np.int_],
    ) -> list[int]:
        # If sap available, then sap
        for opp_unit in opp_unit_positions:
            if np.abs(opp_unit - unit_pos).max() <= self.unit_sap_range:
                dx, dy = opp_unit - unit_pos
                return [5, dx, dy]

        # Otherwise move to the middle of the board
        target = [23, 23] if self.team_id == 0 else [0, 0]
        return [direction_to(unit_pos, target), 0, 0]

    def dump_raw_observations(self) -> None:
        path = PYTHON_SOURCE_ROOT.parent / f"observations_{self.team_id}.json"
        with open(path, "w") as f:
            json.dump(self.all_raw_observations, f)
