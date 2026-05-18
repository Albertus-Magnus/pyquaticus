import numpy as np

# ================================================================
# Discrete action map expected by the competition environment.
# ================================================================
ACTION_MAP = []
for spd in [1.0, 0.5]:
    for hdg in range(180, -180, -45):
        ACTION_MAP.append([spd, hdg])
ACTION_MAP.append([0.0, 0.0])
ACTION_MAP_NP = np.array(ACTION_MAP, dtype=np.float32)

NO_OP = 16

FIELD_WIDTH = 160.0
FIELD_HEIGHT = 80.0
MIDLINE_X = 80.0
WALL_SAFETY = 3.4

BLUE_HOME_CORNERS = [(0.0, 0.0), (0.0, 80.0)]
RED_HOME_CORNERS = [(160.0, 0.0), (160.0, 80.0)]

LANE_Y = {
    "high": 68.0,
    "flex": 40.0,
    "low": 12.0,
}

FULL_ABOUT_FACE = 0
FULL_RIGHT_90 = 2
FULL_STRAIGHT = 4
FULL_LEFT_90 = 6


def angle_wrap_180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def bearing_to_action(rel_bearing_deg: float, half_speed: bool = False) -> int:
    bearing = angle_wrap_180(rel_bearing_deg)
    offset = 8 if half_speed else 0
    candidates = ACTION_MAP_NP[offset:offset + 8, 1]
    diffs = np.abs(((candidates - bearing + 180.0) % 360.0) - 180.0)
    return int(offset + np.argmin(diffs))


def world_bearing_from_pos(from_pos, to_pos, from_heading_deg: float) -> float:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    target_compass = (np.degrees(np.arctan2(dx, dy))) % 360.0
    return angle_wrap_180(target_compass - from_heading_deg)


def sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


class solution:
    """
    Aggressive lane-based controller.

    Design rules:
      - Keep only boundary / wall avoidance.
      - Do not spend actions avoiding teammate or opponent collisions.
      - Attack in spread lanes, then collapse hard on the flag.
      - When a teammate carries, the other agents become pressure / bait.
      - When the enemy carries on our side, chase immediately.
      - Small 2-alive powerplay branch kept, but still direct and aggressive.
    """

    def __init__(self):
        self._mem = {}
        self._team_roles = {}

    def compute_action(self, agent_id: str, full_obs_normalized: dict, full_obs: dict, global_state: dict):
        obs = full_obs[agent_id]
        gs = global_state if global_state is not None else {}

        if agent_id not in self._mem:
            self._mem[agent_id] = {
                "team": None,
                "last_pos": None,
                "stuck_steps": 0,
            }
        mem = self._mem[agent_id]

        if mem["team"] is None:
            mem["team"] = self._detect_team(agent_id, gs)
        team = mem["team"]

        self._maybe_assign_roles(team, gs)
        role = self._role_for(agent_id, team)

        my_pos = self._get_pos(agent_id, gs)
        self._update_stuck(mem, my_pos)

        if self._is_tagged(obs) or self._is_disabled(obs):
            return NO_OP

        # Only hard avoidance we keep: don't go out of bounds.
        dodge = self._wall_dodge(obs)
        if dodge is not None:
            return dodge

        # Hard return if we already hold the flag.
        if self._has_flag(obs):
            return self._carrier_action(obs, gs, agent_id, team, role)

        # Emergency defense if enemy carrier is on our side.
        enemy_carrier = self._pick_enemy_carrier_on_our_side(obs)
        if enemy_carrier is not None and (role == "flex" or self._on_our_side(obs)):
            if self._can_tag(obs) or enemy_carrier["distance"] < 25.0:
                return bearing_to_action(enemy_carrier["bearing"], half_speed=False)

        # If we are in a 2-alive powerplay, stop both agents from diving to the same point.
        if self._in_two_alive_powerplay(team, gs):
            return self._two_alive_powerplay_action(obs, gs, agent_id, team, role)

        # Escort / bait mode once one of us has the flag.
        if self._team_has_flag(obs):
            return self._support_action(obs, gs, agent_id, team, role)

        # Otherwise full attack.
        return self._attack_action(obs, gs, agent_id, team, role)

    # ============================================================
    # State and observation helpers
    # ============================================================
    def _detect_team(self, agent_id, gs):
        pos = self._get_pos(agent_id, gs)
        if pos is not None:
            return "blue" if pos[0] < MIDLINE_X else "red"
        return "blue" if agent_id in ("agent_0", "agent_1", "agent_2") else "red"

    def _team_ids(self, team):
        return ["agent_0", "agent_1", "agent_2"] if team == "blue" else ["agent_3", "agent_4", "agent_5"]

    def _opp_ids(self, team):
        return ["agent_3", "agent_4", "agent_5"] if team == "blue" else ["agent_0", "agent_1", "agent_2"]

    def _maybe_assign_roles(self, team, gs):
        if team in self._team_roles:
            return
        ids = self._team_ids(team)
        pos_list = []
        for aid in ids:
            pos = self._get_pos(aid, gs)
            if pos is not None:
                pos_list.append((aid, pos[1]))
        if len(pos_list) == 3:
            pos_list.sort(key=lambda x: x[1], reverse=True)
            self._team_roles[team] = {
                pos_list[0][0]: "high",
                pos_list[1][0]: "flex",
                pos_list[2][0]: "low",
            }
            return
        self._team_roles[team] = {
            ids[0]: "high",
            ids[1]: "flex",
            ids[2]: "low",
        }

    def _role_for(self, agent_id, team):
        return self._team_roles.get(team, {}).get(agent_id, "flex")

    def _get_pos(self, agent_id, gs):
        key = (agent_id, "pos")
        if key in gs:
            p = gs[key]
            return (float(p[0]), float(p[1]))
        return None

    def _get_heading(self, agent_id, gs):
        key = (agent_id, "heading")
        if key in gs:
            return float(gs[key])
        return 0.0

    def _get_bool(self, agent_id, gs, key_name, default=False):
        key = (agent_id, key_name)
        if key in gs:
            return bool(gs[key])
        return default

    def _is_live_agent(self, agent_id, gs):
        return (not self._get_bool(agent_id, gs, "is_tagged", False)) and (not self._get_bool(agent_id, gs, "is_disabled", False))

    def _nearest_corner(self, pos, team):
        corners = BLUE_HOME_CORNERS if team == "blue" else RED_HOME_CORNERS
        return corners[0] if sqdist(pos, corners[0]) <= sqdist(pos, corners[1]) else corners[1]

    def _team_has_flag(self, obs):
        if self._has_flag(obs):
            return True
        for tm in self._teammates(obs):
            if tm["has_flag"] and not tm["is_tagged"]:
                return True
        return False

    def _has_flag(self, obs):
        return float(obs.get("has_flag", 0.0)) > 0.5

    def _is_tagged(self, obs):
        return float(obs.get("is_tagged", 0.0)) > 0.5

    def _is_disabled(self, obs):
        return float(obs.get("is_disabled", 0.0)) > 0.5

    def _on_our_side(self, obs):
        return float(obs.get("on_side", 0.0)) > 0.5

    def _can_tag(self, obs):
        return float(obs.get("tagging_cooldown", 0.0)) >= 25.0

    def _opponents(self, obs):
        out = []
        for i in range(3):
            name = f"opponent_{i}"
            if (name, "bearing") not in obs:
                continue
            out.append({
                "bearing": float(obs[(name, "bearing")]),
                "distance": float(obs[(name, "distance")]),
                "has_flag": float(obs[(name, "has_flag")]) > 0.5,
                "on_side": float(obs[(name, "on_side")]) > 0.5,
                "is_tagged": float(obs[(name, "is_tagged")]) > 0.5,
                "is_disabled": float(obs[(name, "is_disabled")]) > 0.5,
            })
        return out

    def _teammates(self, obs):
        out = []
        for i in range(2):
            name = f"teammate_{i}"
            if (name, "bearing") not in obs:
                continue
            out.append({
                "bearing": float(obs[(name, "bearing")]),
                "distance": float(obs[(name, "distance")]),
                "has_flag": float(obs[(name, "has_flag")]) > 0.5,
                "on_side": float(obs[(name, "on_side")]) > 0.5,
                "is_tagged": float(obs[(name, "is_tagged")]) > 0.5,
                "is_disabled": float(obs[(name, "is_disabled")]) > 0.5,
            })
        return out

    def _nearest_active_opponent(self, obs, prefer_enemy_side=False):
        candidates = []
        for opp in self._opponents(obs):
            if opp["is_tagged"] or opp["is_disabled"]:
                continue
            if prefer_enemy_side and opp["on_side"]:
                continue
            candidates.append(opp)
        if not candidates:
            return None
        return min(candidates, key=lambda o: o["distance"])

    def _pick_enemy_carrier_on_our_side(self, obs):
        candidates = []
        for opp in self._opponents(obs):
            if opp["has_flag"] and opp["on_side"] and not opp["is_tagged"] and not opp["is_disabled"]:
                candidates.append(opp)
        if not candidates:
            return None
        return min(candidates, key=lambda o: o["distance"])

    def _carrier_teammate(self, obs):
        candidates = []
        for tm in self._teammates(obs):
            if tm["has_flag"] and not tm["is_tagged"]:
                candidates.append(tm)
        if not candidates:
            return None
        return min(candidates, key=lambda t: t["distance"])

    def _count_live_team(self, team, gs):
        return sum(1 for aid in self._team_ids(team) if self._is_live_agent(aid, gs))

    def _count_live_opp(self, team, gs):
        return sum(1 for aid in self._opp_ids(team) if self._is_live_agent(aid, gs))

    # ============================================================
    # Movement primitives
    # ============================================================
    def _update_stuck(self, mem, pos):
        if pos is None:
            return
        last = mem["last_pos"]
        if last is not None and sqdist(pos, last) < 0.20:
            mem["stuck_steps"] += 1
        else:
            mem["stuck_steps"] = 0
        mem["last_pos"] = pos

    def _stuck_escape(self, role):
        if role == "high":
            return FULL_RIGHT_90
        if role == "low":
            return FULL_LEFT_90
        return FULL_ABOUT_FACE

    def _wall_dodge(self, obs):
        worst_b = None
        worst_d = 1e9
        for w in range(4):
            b_key = f"wall_{w}_bearing"
            d_key = f"wall_{w}_distance"
            if b_key not in obs or d_key not in obs:
                continue
            b = float(obs[b_key])
            d = float(obs[d_key])
            if d < WALL_SAFETY and abs(b) < 100.0 and d < worst_d:
                worst_d = d
                worst_b = b
        if worst_b is None:
            return None
        return bearing_to_action(angle_wrap_180(worst_b + 180.0), half_speed=(worst_d < 1.9))

    def _aggressive_staging_point(self, team, role):
        y = LANE_Y[role]
        return (94.0, y) if team == "blue" else (66.0, y)

    def _deep_lane_point(self, team, role):
        y = LANE_Y[role]
        return (118.0, y) if team == "blue" else (42.0, y)

    def _reentry_point(self, team, role):
        y = LANE_Y[role]
        return (90.0, y) if team == "blue" else (70.0, y)

    def _carrier_exit_gates(self, team):
        if team == "blue":
            return [(76.0, y) for y in (8.0, 20.0, 40.0, 60.0, 72.0)]
        return [(84.0, y) for y in (8.0, 20.0, 40.0, 60.0, 72.0)]

    # ============================================================
    # Main behaviors
    # ============================================================
    def _carrier_action(self, obs, gs, agent_id, team, role):
        pos = self._get_pos(agent_id, gs)
        if pos is None:
            return bearing_to_action(float(obs.get("own_home_bearing", 0.0)))

        heading = self._get_heading(agent_id, gs)
        enemy_half = pos[0] > MIDLINE_X if team == "blue" else pos[0] < MIDLINE_X

        if enemy_half:
            gate = self._best_gate(pos, team, gs, self._carrier_exit_gates(team))
            target = gate
        else:
            target = self._nearest_corner(pos, team)

        rel = world_bearing_from_pos(pos, target, heading)
        return bearing_to_action(rel, half_speed=False)

    def _best_gate(self, pos, team, gs, gates):
        opp_positions = [self._get_pos(aid, gs) for aid in self._opp_ids(team)]
        opp_positions = [p for p in opp_positions if p is not None]
        if not opp_positions:
            return min(gates, key=lambda g: sqdist(pos, g))

        best_gate = gates[0]
        best_score = -1e9
        for gate in gates:
            min_opp = min(np.sqrt(sqdist(gate, opp)) for opp in opp_positions)
            travel = np.sqrt(sqdist(pos, gate))
            score = min_opp - 0.22 * travel
            if score > best_score:
                best_score = score
                best_gate = gate
        return best_gate

    def _support_action(self, obs, gs, agent_id, team, role):
        mem = self._mem[agent_id]
        if mem["stuck_steps"] >= 7:
            return self._stuck_escape(role)

        carrier_tm = self._carrier_teammate(obs)
        threat = self._nearest_active_opponent(obs, prefer_enemy_side=True)

        # In enemy territory, become bait / pressure. No dodge blending.
        if not self._on_our_side(obs):
            if threat is not None:
                if carrier_tm is not None:
                    bait_bearing = angle_wrap_180(0.85 * threat["bearing"] + 0.15 * carrier_tm["bearing"])
                    return bearing_to_action(bait_bearing)
                return bearing_to_action(threat["bearing"])
            return bearing_to_action(float(obs.get("opponent_home_bearing", 0.0)))

        # On our side, get back across unless we are offsetting as screen.
        if role == "flex":
            return bearing_to_action(float(obs.get("opponent_home_bearing", 0.0)))

        if carrier_tm is not None:
            offset = 55.0 if role == "high" else -55.0
            return bearing_to_action(angle_wrap_180(carrier_tm["bearing"] + offset))

        return bearing_to_action(float(obs.get("opponent_home_bearing", 0.0)))

    def _attack_action(self, obs, gs, agent_id, team, role):
        mem = self._mem[agent_id]
        if mem["stuck_steps"] >= 7:
            return self._stuck_escape(role)

        pos = self._get_pos(agent_id, gs)
        if pos is None:
            return bearing_to_action(float(obs.get("opponent_home_bearing", 0.0)))

        heading = self._get_heading(agent_id, gs)
        opp_flag_b = float(obs.get("opponent_home_bearing", 0.0))
        opp_flag_d = float(obs.get("opponent_home_distance", 999.0))
        lane_y = LANE_Y[role]

        if self._on_our_side(obs):
            target = self._aggressive_staging_point(team, role)
            rel = world_bearing_from_pos(pos, target, heading)
            return bearing_to_action(rel)

        # Stay spread until close enough, then commit directly.
        if opp_flag_d > 18.0 and abs(pos[1] - lane_y) > 8.0:
            target = self._deep_lane_point(team, role)
            rel = world_bearing_from_pos(pos, target, heading)
            return bearing_to_action(rel)

        # If we can tag someone near the flag lane, take the fight.
        threat = self._nearest_active_opponent(obs, prefer_enemy_side=True)
        if threat is not None and threat["distance"] < 11.0 and self._can_tag(obs):
            return bearing_to_action(threat["bearing"])

        # Otherwise go straight through traffic to the flag.
        return bearing_to_action(opp_flag_b)

    # ============================================================
    # Small special case: 2 alive while we still have advantage
    # ============================================================
    def _in_two_alive_powerplay(self, team, gs):
        our_live = self._count_live_team(team, gs)
        opp_live = self._count_live_opp(team, gs)
        return our_live == 2 and our_live > opp_live

    def _two_alive_powerplay_action(self, obs, gs, agent_id, team, role):
        if self._has_flag(obs):
            return self._carrier_action(obs, gs, agent_id, team, role)

        enemy_carrier = self._pick_enemy_carrier_on_our_side(obs)
        if enemy_carrier is not None:
            return bearing_to_action(enemy_carrier["bearing"])

        if self._team_has_flag(obs):
            carrier_tm = self._carrier_teammate(obs)
            if carrier_tm is not None:
                if role == "flex":
                    return bearing_to_action(float(obs.get("own_home_bearing", 0.0)))
                offset = 45.0 if role == "high" else -45.0
                return bearing_to_action(angle_wrap_180(carrier_tm["bearing"] + offset))
            return bearing_to_action(float(obs.get("own_home_bearing", 0.0)))

        if role == "flex":
            own_home_b = float(obs.get("own_home_bearing", 0.0))
            if self._on_our_side(obs):
                return bearing_to_action(own_home_b)
            return bearing_to_action(angle_wrap_180(own_home_b + 180.0))

        return self._attack_action(obs, gs, agent_id, team, role)
