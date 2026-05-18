import numpy as np


ACTION_MAP = []
for spd in [1.0, 0.5]:
    for hdg in range(180, -180, -45):
        ACTION_MAP.append([spd, hdg])
ACTION_MAP.append([0.0, 0.0])
ACTION_MAP_NP = np.array(ACTION_MAP, dtype=np.float32)

NO_OP = 16
FULL_ABOUT_FACE = 0
FULL_RIGHT_90 = 2
FULL_STRAIGHT = 4
FULL_LEFT_90 = 6

FIELD_WIDTH = 160.0
FIELD_HEIGHT = 80.0
MIDLINE_X = 80.0
WALL_SAFETY = 3.0

BLUE_DEFAULT_FLAG = (0.0, 40.0)
RED_DEFAULT_FLAG = (160.0, 40.0)
BLUE_IDS = ("agent_0", "agent_1", "agent_2")
RED_IDS = ("agent_3", "agent_4", "agent_5")
BLUE_ENTRY_X = 93.0
RED_ENTRY_X = 67.0
BLUE_DEEP_X = 118.0
RED_DEEP_X = 42.0
BLUE_SAFETY = (57.0, 40.0)
RED_SAFETY = (103.0, 40.0)

LANE_Y = {
    "high": 68.0,
    "flex": 40.0,
    "low": 12.0,
}

BLUE_SAFE_CORNERS = [(10.0, 10.0), (10.0, 70.0)]
RED_SAFE_CORNERS = [(150.0, 10.0), (150.0, 70.0)]


def angle_wrap_180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bounded_point(p):
    return (clamp(float(p[0]), 0.0, FIELD_WIDTH), clamp(float(p[1]), 0.0, FIELD_HEIGHT))


def sqdist(a, b):
    if a is None or b is None:
        return 1e18
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a, b):
    return float(np.sqrt(sqdist(a, b)))


def lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vec_mul(a, k):
    return (a[0] * k, a[1] * k)


def vec_unit(a):
    n = float(np.hypot(a[0], a[1]))
    if n < 1e-8:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def vec_perp(a):
    return (-a[1], a[0])


def bearing_to_action(rel_bearing_deg, half_speed=False):
    bearing = angle_wrap_180(rel_bearing_deg)
    offset = 8 if half_speed else 0
    candidates = ACTION_MAP_NP[offset:offset + 8, 1]
    diffs = np.abs(((candidates - bearing + 180.0) % 360.0) - 180.0)
    return int(offset + np.argmin(diffs))


def world_bearing_from_pos(from_pos, to_pos, from_heading_deg):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    target_compass = (np.degrees(np.arctan2(dx, dy))) % 360.0
    return angle_wrap_180(target_compass - from_heading_deg)


class solution:
    """
    Scoring-first MCTF controller.

    Neutral shape is two live scoring threats plus one safety/interceptor. The
    controller deliberately ignores teammate traffic and only spends actions on
    wall/bounds recovery, carrier conversion, and useful defensive stops.
    """

    def __init__(self):
        self._mem = {}
        self._lane_roles = {}

    def compute_action(self, agent_id: str, full_obs_normalized: dict, full_obs: dict, global_state: dict):
        obs = full_obs[agent_id]
        gs = global_state if global_state is not None else {}

        mem = self._mem.setdefault(agent_id, {"team": None, "last_pos": None, "stuck_steps": 0})
        if mem["team"] is None:
            mem["team"] = self._detect_team(agent_id, gs)
        team = mem["team"]

        self._assign_lanes(team, gs)
        role = self._role(agent_id, team)

        pos = self._get_pos(agent_id, gs)
        self._update_stuck(mem, pos)

        if self._is_tagged(obs) or self._is_disabled(obs):
            return NO_OP

        dodge = self._wall_dodge(obs)
        if dodge is not None:
            return dodge

        target, half_speed, intent = self._target_for(agent_id, team, role, obs, gs)

        if mem["stuck_steps"] >= 9 and intent not in {"carrier", "chase"}:
            return self._stuck_escape(role)

        if target is None or pos is None:
            return self._fallback(obs)

        heading = self._get_heading(agent_id, gs)
        rel = world_bearing_from_pos(pos, target, heading)
        return bearing_to_action(rel, half_speed=half_speed)

    # ------------------------------------------------------------------
    # Top-level tactics
    # ------------------------------------------------------------------
    def _target_for(self, agent_id, team, role, obs, gs):
        live_us = self._live_team(team, gs)
        live_them = self._live_opp(team, gs)
        our_carrier = self._team_carrier(team, gs)
        enemy_carrier = self._enemy_carrier(team, gs)

        if self._has_flag(obs) or agent_id == our_carrier:
            return self._carrier_target(agent_id, team, gs), False, "carrier"

        if len(live_us) <= 2:
            return self._powerplay_target(agent_id, team, role, gs, live_us, our_carrier, enemy_carrier)

        if enemy_carrier is not None and self._pos_on_our_side(self._get_pos(enemy_carrier, gs), team):
            return self._enemy_carrier_target(agent_id, team, role, gs, live_us, enemy_carrier)

        if our_carrier is not None:
            return self._support_carrier_target(agent_id, team, role, gs, live_us, our_carrier)

        return self._neutral_target(agent_id, team, role, gs, live_us, live_them)

    def _neutral_target(self, agent_id, team, role, gs, live_us, live_them):
        enemy_flag = self._flag_home(team, gs, enemy=True)
        home_flag = self._flag_home(team, gs, enemy=False)
        threats = self._enemy_threats_on_our_side(team, gs)
        score_diff = self._score(team, gs) - self._opp_score(team, gs)

        attackers = self._pick_attackers(team, gs, live_us)
        if agent_id in attackers:
            slot = attackers.index(agent_id)
            return self._attack_target(agent_id, team, role, gs, slot), False, "attack"

        if threats:
            threat = min(threats, key=lambda aid: dist(self._get_pos(aid, gs), home_flag))
            return self._lane_block(self._get_pos(threat, gs), home_flag, self._get_pos(agent_id, gs), 0.48, self._role_lateral(role) * 0.5), False, "safety"

        if score_diff <= -2 and len(live_them) >= 2:
            return self._attack_target(agent_id, team, role, gs, 2), False, "attack"

        return self._safety_anchor(team, gs), True, "safety"

    def _support_carrier_target(self, agent_id, team, role, gs, live_us, carrier_id):
        carrier_pos = self._get_pos(carrier_id, gs)
        if carrier_pos is None:
            return self._attack_target(agent_id, team, role, gs, 1), False, "attack"

        helpers = [aid for aid in live_us if aid != carrier_id]
        if not helpers:
            return self._safety_anchor(team, gs), True, "safety"

        closest_helper = min(helpers, key=lambda aid: dist(self._get_pos(aid, gs), carrier_pos))
        if agent_id == closest_helper:
            return self._screen_carrier(agent_id, team, gs, carrier_id), False, "escort"

        enemy_flag = self._flag_home(team, gs, enemy=True)
        nearest_threat = self._nearest_live_opp_to_point(team, gs, carrier_pos)
        if nearest_threat is not None and dist(nearest_threat, carrier_pos) < 18.0:
            return self._harass_between(nearest_threat, carrier_pos, role), False, "harass"
        if dist(self._get_pos(agent_id, gs), enemy_flag) < 28.0:
            return enemy_flag, False, "attack"
        return self._carrier_lane_support(team, gs, carrier_id, role), False, "escort"

    def _enemy_carrier_target(self, agent_id, team, role, gs, live_us, enemy_carrier):
        carrier_pos = self._get_pos(enemy_carrier, gs)
        home = self._flag_home(team, gs, enemy=False)
        chaser = min(live_us, key=lambda aid: dist(self._get_pos(aid, gs), carrier_pos))
        if agent_id == chaser:
            return carrier_pos, False, "chase"

        remaining = [aid for aid in live_us if aid != chaser]
        blocker = min(remaining, key=lambda aid: dist(self._get_pos(aid, gs), self._lane_block(carrier_pos, home, self._get_pos(aid, gs), 0.58, self._role_lateral(role))))
        if agent_id == blocker:
            return self._lane_block(carrier_pos, home, self._get_pos(agent_id, gs), 0.58, self._role_lateral(role)), False, "block"

        return self._safety_anchor(team, gs), True, "safety"

    def _powerplay_target(self, agent_id, team, role, gs, live_us, our_carrier, enemy_carrier):
        if agent_id not in live_us:
            return None, True, "idle"

        home = self._flag_home(team, gs, enemy=False)
        enemy_flag = self._flag_home(team, gs, enemy=True)

        if our_carrier is not None:
            if agent_id == our_carrier:
                return self._carrier_target(agent_id, team, gs), False, "carrier"
            return self._screen_carrier(agent_id, team, gs, our_carrier), True, "escort"

        if enemy_carrier is not None and self._pos_on_our_side(self._get_pos(enemy_carrier, gs), team):
            carrier_pos = self._get_pos(enemy_carrier, gs)
            chaser = min(live_us, key=lambda aid: dist(self._get_pos(aid, gs), carrier_pos))
            if agent_id == chaser:
                return carrier_pos, False, "chase"
            return self._lane_block(carrier_pos, home, self._get_pos(agent_id, gs), 0.60, self._role_lateral(role) * 0.5), True, "block"

        probe = min(live_us, key=lambda aid: dist(self._get_pos(aid, gs), enemy_flag))
        if agent_id == probe:
            pos = self._get_pos(agent_id, gs)
            if pos is not None and self._pos_enemy_half(pos, team):
                return enemy_flag, False, "attack"
            return self._entry_point(team, role), True, "attack"

        return lerp(home, enemy_flag, 0.28), True, "safety"

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    def _attack_target(self, agent_id, team, role, gs, slot):
        pos = self._get_pos(agent_id, gs)
        enemy_flag = self._flag_home(team, gs, enemy=True)
        lane_y = self._lane_y(role, slot)
        if pos is None:
            return self._entry_point(team, role)

        if self._pos_on_our_side(pos, team):
            return self._entry_point(team, role)

        if dist(pos, enemy_flag) <= 18.0:
            return enemy_flag

        x = BLUE_DEEP_X if team == "blue" else RED_DEEP_X
        if abs(pos[1] - lane_y) > 7.0:
            return (x, lane_y)

        return enemy_flag

    def _carrier_target(self, agent_id, team, gs):
        pos = self._get_pos(agent_id, gs)
        if pos is None:
            return self._home_corner(team, (0.0, 40.0))

        if not self._pos_enemy_half(pos, team):
            return self._home_corner(team, pos)

        gates = self._exit_gates(team)
        threats = [self._get_pos(aid, gs) for aid in self._live_opp(team, gs)]
        threats = [p for p in threats if p is not None]
        if not threats:
            return min(gates, key=lambda g: sqdist(pos, g))

        best_gate = gates[0]
        best_score = -1e9
        for gate in gates:
            nearest_threat = min(dist(gate, p) for p in threats)
            travel = dist(pos, gate)
            y_keep = -0.06 * abs(gate[1] - pos[1])
            edge_bonus = 1.6 if gate[1] <= 18.0 or gate[1] >= 62.0 else 0.0
            score = 1.35 * nearest_threat - 0.20 * travel + y_keep + edge_bonus
            if score > best_score:
                best_score = score
                best_gate = gate
        return best_gate

    def _screen_carrier(self, agent_id, team, gs, carrier_id):
        pos = self._get_pos(agent_id, gs)
        carrier_pos = self._get_pos(carrier_id, gs)
        if carrier_pos is None:
            return self._safety_anchor(team, gs)

        if self._pos_enemy_half(carrier_pos, team):
            target = self._carrier_target(carrier_id, team, gs)
        else:
            target = self._home_corner(team, carrier_pos)

        route = vec_unit(vec_sub(target, carrier_pos))
        lateral = vec_perp(route)
        threat = self._nearest_live_opp_to_point(team, gs, carrier_pos)
        side = self._side(threat if threat is not None else pos, carrier_pos, target)
        return bounded_point(vec_add(vec_add(carrier_pos, vec_mul(route, 7.0)), vec_mul(lateral, -side * 4.5)))

    def _carrier_lane_support(self, team, gs, carrier_id, role):
        carrier_pos = self._get_pos(carrier_id, gs)
        if carrier_pos is None:
            return self._safety_anchor(team, gs)
        target = self._carrier_target(carrier_id, team, gs)
        base = lerp(carrier_pos, target, 0.50)
        return bounded_point((base[0], base[1] + self._role_lateral(role) * 0.65))

    def _harass_between(self, threat_pos, carrier_pos, role):
        base = lerp(threat_pos, carrier_pos, 0.35)
        return bounded_point((base[0], base[1] + self._role_lateral(role) * 0.35))

    def _lane_block(self, carrier_pos, home, blocker_pos, ratio, lateral):
        if carrier_pos is None:
            return home
        base = lerp(carrier_pos, home, ratio)
        route = vec_unit(vec_sub(home, carrier_pos))
        side = self._side(blocker_pos, carrier_pos, home)
        return bounded_point(vec_add(base, vec_mul(vec_perp(route), side * abs(lateral))))

    def _safety_anchor(self, team, gs):
        threats = self._enemy_threats_on_our_side(team, gs)
        home = self._flag_home(team, gs, enemy=False)
        if threats:
            threat = min(threats, key=lambda aid: dist(self._get_pos(aid, gs), home))
            return self._lane_block(self._get_pos(threat, gs), home, None, 0.42, 0.0)
        return BLUE_SAFETY if team == "blue" else RED_SAFETY

    # ------------------------------------------------------------------
    # Team/state helpers
    # ------------------------------------------------------------------
    def _team_ids(self, team):
        return list(BLUE_IDS if team == "blue" else RED_IDS)

    def _opp_ids(self, team):
        return list(RED_IDS if team == "blue" else BLUE_IDS)

    def _detect_team(self, agent_id, gs):
        pos = self._get_pos(agent_id, gs)
        if pos is not None:
            return "blue" if pos[0] < MIDLINE_X else "red"
        return "blue" if agent_id in BLUE_IDS else "red"

    def _assign_lanes(self, team, gs):
        if team in self._lane_roles:
            return
        ids = self._team_ids(team)
        starts = []
        for aid in ids:
            pos = self._get_pos(aid, gs)
            if pos is not None:
                starts.append((aid, pos[1]))
        if len(starts) == 3:
            starts.sort(key=lambda item: item[1], reverse=True)
            self._lane_roles[team] = {
                starts[0][0]: "high",
                starts[1][0]: "flex",
                starts[2][0]: "low",
            }
        else:
            self._lane_roles[team] = {ids[0]: "high", ids[1]: "flex", ids[2]: "low"}

    def _role(self, agent_id, team):
        return self._lane_roles.get(team, {}).get(agent_id, "flex")

    def _pick_attackers(self, team, gs, live_us):
        enemy_flag = self._flag_home(team, gs, enemy=True)
        ranked = sorted(live_us, key=lambda aid: dist(self._get_pos(aid, gs), enemy_flag))
        return ranked[: min(2, len(ranked))]

    def _live_team(self, team, gs):
        return [aid for aid in self._team_ids(team) if self._is_live(aid, gs)]

    def _live_opp(self, team, gs):
        return [aid for aid in self._opp_ids(team) if self._is_live(aid, gs)]

    def _is_live(self, agent_id, gs):
        return (
            self._get_pos(agent_id, gs) is not None
            and not self._get_bool(agent_id, gs, "is_tagged", False)
            and not self._get_bool(agent_id, gs, "is_disabled", False)
            and not self._get_bool(agent_id, gs, "oob", False)
            and self._get_bool(agent_id, gs, "alive", True)
        )

    def _team_carrier(self, team, gs):
        for aid in self._team_ids(team):
            if self._get_bool(aid, gs, "has_flag", False):
                return aid
        return None

    def _enemy_carrier(self, team, gs):
        for aid in self._opp_ids(team):
            if self._get_bool(aid, gs, "has_flag", False):
                return aid
        return None

    def _enemy_threats_on_our_side(self, team, gs):
        out = []
        for aid in self._live_opp(team, gs):
            pos = self._get_pos(aid, gs)
            if self._pos_on_our_side(pos, team):
                out.append(aid)
        return out

    def _nearest_live_opp_to_point(self, team, gs, point):
        if point is None:
            return None
        positions = [self._get_pos(aid, gs) for aid in self._live_opp(team, gs)]
        positions = [p for p in positions if p is not None]
        if not positions:
            return None
        return min(positions, key=lambda p: dist(p, point))

    def _get_pos(self, agent_id, gs):
        if agent_id is None:
            return None
        key = (agent_id, "pos")
        if key not in gs:
            return None
        p = gs[key]
        return (float(p[0]), float(p[1]))

    def _get_heading(self, agent_id, gs):
        return float(gs.get((agent_id, "heading"), 0.0))

    def _get_bool(self, agent_id, gs, field, default=False):
        if agent_id is None:
            return bool(default)
        return bool(gs.get((agent_id, field), default))

    def _score(self, team, gs):
        return int(gs.get("blue_team_score", 0) if team == "blue" else gs.get("red_team_score", 0))

    def _opp_score(self, team, gs):
        return int(gs.get("red_team_score", 0) if team == "blue" else gs.get("blue_team_score", 0))

    def _flag_home(self, team, gs, enemy):
        if team == "blue":
            key = "red_flag_home" if enemy else "blue_flag_home"
            fallback = RED_DEFAULT_FLAG if enemy else BLUE_DEFAULT_FLAG
        else:
            key = "blue_flag_home" if enemy else "red_flag_home"
            fallback = BLUE_DEFAULT_FLAG if enemy else RED_DEFAULT_FLAG
        p = gs.get(key, fallback)
        return (float(p[0]), float(p[1]))

    def _pos_on_our_side(self, pos, team):
        if pos is None:
            return False
        return pos[0] <= MIDLINE_X if team == "blue" else pos[0] >= MIDLINE_X

    def _pos_enemy_half(self, pos, team):
        if pos is None:
            return False
        return pos[0] > MIDLINE_X if team == "blue" else pos[0] < MIDLINE_X

    def _entry_point(self, team, role):
        return ((BLUE_ENTRY_X if team == "blue" else RED_ENTRY_X), LANE_Y[role])

    def _exit_gates(self, team):
        x = 76.0 if team == "blue" else 84.0
        return [(x, y) for y in (8.0, 18.0, 30.0, 40.0, 52.0, 64.0, 72.0)]

    def _home_corner(self, team, pos):
        corners = BLUE_SAFE_CORNERS if team == "blue" else RED_SAFE_CORNERS
        return min(corners, key=lambda c: sqdist(c, pos))

    def _lane_y(self, role, slot):
        if role in LANE_Y:
            return LANE_Y[role]
        return (68.0, 12.0, 40.0)[slot % 3]

    def _role_lateral(self, role):
        if role == "high":
            return 10.0
        if role == "low":
            return -10.0
        return 0.0

    def _side(self, p, a, b):
        if p is None or a is None or b is None:
            return 1.0
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        apx = p[0] - a[0]
        apy = p[1] - a[1]
        return 1.0 if (abx * apy - aby * apx) >= 0.0 else -1.0

    # ------------------------------------------------------------------
    # Observation helpers and movement safety
    # ------------------------------------------------------------------
    def _has_flag(self, obs):
        return float(obs.get("has_flag", 0.0)) > 0.5

    def _is_tagged(self, obs):
        return float(obs.get("is_tagged", 0.0)) > 0.5

    def _is_disabled(self, obs):
        return float(obs.get("is_disabled", 0.0)) > 0.5

    def _fallback(self, obs):
        enemy_carrier = self._obs_enemy_carrier(obs)
        if enemy_carrier is not None:
            return bearing_to_action(enemy_carrier["bearing"], half_speed=False)
        if self._has_flag(obs):
            return bearing_to_action(float(obs.get("own_home_bearing", 0.0)), half_speed=False)
        return bearing_to_action(float(obs.get("opponent_home_bearing", 0.0)), half_speed=False)

    def _obs_enemy_carrier(self, obs):
        candidates = []
        for i in range(3):
            name = f"opponent_{i}"
            if (name, "bearing") not in obs:
                continue
            if float(obs.get((name, "has_flag"), 0.0)) <= 0.5:
                continue
            if float(obs.get((name, "is_tagged"), 0.0)) > 0.5:
                continue
            candidates.append({
                "bearing": float(obs[(name, "bearing")]),
                "distance": float(obs.get((name, "distance"), 999.0)),
            })
        if not candidates:
            return None
        return min(candidates, key=lambda item: item["distance"])

    def _wall_dodge(self, obs):
        worst_bearing = None
        worst_distance = 1e9
        for idx in range(4):
            b_key = f"wall_{idx}_bearing"
            d_key = f"wall_{idx}_distance"
            if b_key not in obs or d_key not in obs:
                continue
            bearing = float(obs[b_key])
            distance = float(obs[d_key])
            if distance < WALL_SAFETY and abs(bearing) < 105.0 and distance < worst_distance:
                worst_bearing = bearing
                worst_distance = distance
        if worst_bearing is None:
            return None
        return bearing_to_action(angle_wrap_180(worst_bearing + 180.0), half_speed=worst_distance < 1.8)

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
