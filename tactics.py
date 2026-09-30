"""Coordinated front concentration for Nova v6."""
from collections import defaultdict
from time import perf_counter


def consolidate(board, plans, projected, reach, flagreach, garrisons, deadline):
    """Concentrate nearby offensive groups without stealing defenders.

    v5 only merged groups that were already stalled, so in practice almost no
    regrouping occurred.  v6 may rally advancing groups onto a shared adjacent
    tile as long as no group moves farther from its own goal and all garrison
    floors remain valid. Capture/escort plans are excluded because an F may be
    relying on those troops.
    """
    g = board
    changed = 0
    for _ in range(2):
        if perf_counter() >= deadline:
            break
        candidates = defaultdict(list)
        for i, (origin, dest, count, goal, value, role) in enumerate(plans):
            if not count or role not in ('advance', 'intercept', 'assault'):
                continue
            if origin == goal:
                continue
            for q in g.adj[origin]:
                if q == origin:
                    continue
                # Never retreat from the group's own mission just to merge.
                if g.dist[q][goal] <= g.dist[origin][goal]:
                    candidates[q].append(i)
        best = None
        for q, indices in candidates.items():
            if len(indices) < 2:
                continue
            count = sum(plans[i][2] for i in indices)
            removed = defaultdict(int)
            for i in indices:
                removed[plans[i][1]] += plans[i][2]
            at_q = projected[q] - removed[q] + count
            # Rally only if the merged stack is genuinely viable, not merely
            # equal to hostile reach. A small margin is what creates local
            # superiority rather than two independent losing fights.
            margin=1+int(reach[q]>=6)
            if at_q < reach[q] + bool(flagreach[q]) + margin:
                continue
            if any(projected[p] - amount + (count if p == q else 0) < garrisons.get(p, 0)
                   for p, amount in removed.items()):
                continue
            # Every group retains its own goal. Do not buy one breakthrough by
            # silently abandoning another capture that its F is relying on.
            utility = sum(plans[i][2] * plans[i][4] /
                          (2 + g.dist[q][plans[i][3]]) for i in indices) / count
            key = (utility, count, -g.key(q))
            if best is None or key > best[0]:
                best = key, q, indices
        if best is None:
            break
        _, q, indices = best
        for i in indices:
            plan = plans[i]
            projected[plan[1]] -= plan[2]
            plan[1] = q
            projected[q] += plan[2]
        changed += 1
    return changed
