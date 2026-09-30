"""Bounded final-turn search using public observations, not replay information.

Opponent deployments are legal stress cases, not calibrated probabilities.
The ordinary policy supplies production and TELE; this layer adjusts local moves.
"""
from collections import Counter
from time import perf_counter


class OwnershipHistory:
    def __init__(self):
        self.turn = 0
        self.balance = Counter()

    def observe(self, turn, buildings, team, enemy):
        # TURN t contains the result of t-1. Count each observation once.
        if turn <= 1 or turn - 1 <= self.turn:
            return
        for p, b in buildings.items():
            self.balance[p] += (b['owner'] == team) - (b['owner'] == enemy)
        self.turn = turn - 1

    def interval(self, buildings, known, board):
        low = high = 0
        # Equal-score mirrored buildings must share the same uncertainty.
        pairs = Counter()
        for p, count in self.balance.items():
            pairs[min(p, board.mirror(p))] += count
        for p, count in pairs.items():
            if p in known:
                a = b = known[p]
            elif buildings[p]['kind'] == 'PLAZA':
                a = b = 3
            else:
                a, b = (2, 4) if 5 <= board.xy(p)[0] <= 9 else (1, 2)
            low += min(a * count, b * count)
            high += max(a * count, b * count)
        return low, high


def finish(board, buildings, team, enemy, ours, hostile, enemy_money,
           enemy_sites, enemy_cost, budget, plans, flagplans, projected,
           teleport, score, occupation, deadline):
    """Improve the existing legal deployment; stop with a valid best-so-far plan."""
    g = board
    cells = list(buildings)
    values = {p: score(p) for p in cells}
    # Stable score order is also emitted as the actual PRIORITY command.
    priority = sorted(cells, key=lambda p: (-values[p], g.key(p)))
    ew = hostile['W'].copy()
    ef = hostile['F'].copy()
    # Explicit forecast: opponent spends on W at its nearest relevant spawn.
    relevant = [p for p in cells if any(ours['F'][q] or ef[q] for q in g.adj[p])]
    site = min(enemy_sites, key=lambda s: (min((g.dist[s][p] for p in relevant), default=0), -g.key(s)))
    ew[site] += enemy_money // enemy_cost
    enemy_budget = enemy_money % enemy_cost
    scenarios = [(ew, ef, enemy_budget)]
    # Each focus case moves only original units once. Unlike a full reach sum,
    # the same enemy W is never simultaneously present at several buildings.
    focus = sorted((p for p in cells if any(ew[q] or ef[q] for q in g.adj[p])),
                   key=lambda p: (-values[p], g.key(p)))
    for p in focus:
        w, f = ew.copy(), ef.copy()
        for q in g.adj[p]:
            if q != p:
                w[p] += ew[q]; w[q] -= ew[q]
                f[p] += ef[q]; f[q] -= ef[q]
        scenarios.append((w, f, enemy_budget))
    # A dispersed capture scenario complements concentrated attacks.
    w, f = [0] * g.n, [0] * g.n
    for origin in range(g.n):
        options = [q for q in g.adj[origin] if q in buildings]
        if ef[origin]:
            dest = max(options, key=lambda q: (values[q] * (buildings[q]['owner'] != enemy), -projected[q], -g.key(q)), default=origin)
            f[dest] += ef[origin]
        if ew[origin]:
            dest = max(options, key=lambda q: (values[q] * (bool(ours['F'][q]) + bool(ef[q]) + (buildings[q]['owner'] == enemy)), -g.key(q)), default=origin)
            w[dest] += ew[origin]
    scenarios.append((w, f, enemy_budget))
    # Last-turn hospital/base F production and station TELE are also legal
    # immediate threats. Keep their resource and original-unit pools separate.
    if enemy_money >= 5:
        for spawn in enemy_sites:
            for dest in g.adj[spawn]:
                if dest not in buildings or buildings[dest]['owner'] == enemy:
                    continue
                w, f = hostile['W'].copy(), ef.copy()
                w[spawn] += (enemy_money - 5) // enemy_cost
                f[dest] += 1
                scenarios.append((w, f, (enemy_money - 5) % enemy_cost))
    stations = [p for p in cells if buildings[p]['owner'] == enemy and buildings[p]['kind'] == 'STATION']
    for src in stations:
        for dest in stations:
            if src == dest or not ew[src]: continue
            w = ew.copy(); amount = min(5, ew[src])
            w[src] -= amount; w[dest] += amount
            scenarios.append((w, ef, enemy_budget))
    final_f = [0] * g.n
    for _, q, count, _ in flagplans:
        final_f[q] += count

    def evaluate():
        rewards = []
        for enemy_w, enemy_f, remaining_money in scenarios:
            owners = {}; flags = {}; halls = Counter(); libraries = set()
            for p in cells:
                owner = buildings[p]['owner']
                a, b = final_f[p], enemy_f[p]
                if projected[p] < enemy_w[p]:
                    if a and owner == team: owner = 'N'
                    a = 0
                elif projected[p] > enemy_w[p]:
                    if b and owner == enemy: owner = 'N'
                    b = 0
                owners[p] = owner; flags[p] = (a, b)
                if buildings[p]['kind'] == 'HALL': halls[owner] += 1
                if buildings[p]['kind'] == 'LIBRARY': libraries.add(owner)
            cash = {team: min(40, budget + 10 + 2 * halls[team]),
                    enemy: min(40, remaining_money + 10 + 2 * halls[enemy])}
            for t, index in ((team, 0), (enemy, 1)):
                for p in priority:
                    a, b = flags[p][index], flags[p][1-index]
                    if not a or b or owners[p] == t: continue
                    cost = max(1, (4 if buildings[p]['kind'] == 'PLAZA' else 2) - (t in libraries))
                    if cash[t] < cost: continue
                    cash[t] -= cost
                    owners[p] = t if owners[p] == 'N' else 'N'
            margin = sum(values[p] * ((owners[p] == team) - (owners[p] == enemy)) for p in cells)
            lo, hi = occupation[0] + margin, occupation[1] + margin
            if margin > 0: outcome = 1
            elif margin < 0: outcome = -1
            elif lo > 0: outcome = 1
            elif hi < 0: outcome = -1
            elif lo == hi == 0:
                # W cancellation preserves the global W count difference.
                unit_gap = 3 * (sum(projected) - sum(enemy_w))
                for p in range(g.n):
                    if projected[p] >= enemy_w[p]:
                        unit_gap += 5 * final_f[p] + 2 * ours['S'][p]
                    if enemy_w[p] >= projected[p]:
                        unit_gap -= 5 * enemy_f[p] + 2 * hostile['S'][p]
                outcome = (unit_gap > 0) - (unit_gap < 0)
            else: outcome = 0  # Unknown score interval: no invented tiebreak win.
            rewards.append(100 * outcome + margin)
        # Prefer winning scenarios; preserve guaranteed wins when available.
        return min(rewards) * .15 + sum(rewards) / len(rewards)

    baseline = evaluate(); best = baseline; changes = 0
    # F first lets W subsequently reinforce a newly selected capture.
    # Include all adjacent buildings, even if not the old assigned objective.
    for sweep in range(2):
        for plan in flagplans:
            if perf_counter() >= deadline: break
            origin, old, count, goal = plan
            choice = old
            final_f[old] -= count
            for q in g.adj[origin]:
                final_f[q] += count
                trial = evaluate()
                final_f[q] -= count
                if trial > best + 1e-9: best, choice = trial, q
            final_f[choice] += count
            if choice != old: changes += 1
            plan[1] = choice
        for plan in plans:
            if perf_counter() >= deadline: break
            origin, old, count, *_ = plan
            if not count: continue
            choice = old
            projected[old] -= count
            for q in g.adj[origin]:
                projected[q] += count
                trial = evaluate()
                projected[q] -= count
                if trial > best + 1e-9: best, choice = trial, q
            projected[choice] += count
            if choice != old: changes += 1
            plan[1] = choice
        if perf_counter() >= deadline: break
    return priority, {'scenarios': len(scenarios), 'changes': changes,
                      'baseline': baseline, 'utility': best,
                      'occupation_interval': occupation}
