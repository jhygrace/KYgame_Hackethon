"""Public-information opponent model and confidence-gated local response.

The model is the supplied Nova v4, not an oracle. Unknown enemy score knowledge
stays unknown; actual post-combat observations determine whether to trust it.
"""
from collections import defaultdict
from time import perf_counter
from reference import Nova as ReferenceNova


class OpponentModel:
    def __init__(self, terrain, bases, team, buildings):
        enemy = 'K' if team == 'Y' else 'Y'
        self.bot = ReferenceNova(terrain, bases, enemy, buildings)
        self.team = enemy
        self.seen = set()
        self.expected = None
        self.confidence = .5
        self.prediction = None

    def forecast(self, turn, money, enemy_money, units, buildings, known):
        g = self.bot.board
        actual = {k: [0] * g.n for k in 'FWS'}
        for t, k, x, y, c in units:
            if t == self.team:
                actual[k][g.cell(x, y)] += c
        if self.expected is not None:
            accuracy = {}
            for kind in 'FW':
                a, b = actual[kind], self.expected[kind]
                accuracy[kind] = 1 - sum(abs(x-y) for x, y in zip(a,b)) / max(1, sum(a)+sum(b))
            quality = .6 * accuracy['W'] + .4 * accuracy['F']
            self.confidence = .6 * self.confidence + .4 * quality

        watchers = [(x,y) for _,x,y,kind,owner,_,_ in buildings
                    if kind == 'WATCH' and owner == self.team]
        scouts = [g.xy(p) for p,c in enumerate(actual['S']) if c]
        filtered = []
        for ident,x,y,kind,owner,stage,_ in buildings:
            p = g.cell(x,y)
            if (owner == self.team or any(actual[k][p] for k in 'FWS') or
                any(max(abs(x-xx),abs(y-yy)) <= 3 for xx,yy in watchers) or
                any(max(abs(x-xx),abs(y-yy)) <= 2 for xx,yy in scouts)):
                self.seen.add(p)
            value = known.get(p, -1) if p in self.seen else -1
            filtered.append((ident,x,y,kind,owner,stage,value))
        # Avoid a second terminal search; our own endgame layer handles t160.
        commands = self.bot.decide(min(turn,159), enemy_money, money, units, filtered)
        stationary, prediction = project(g, self.bot.base, actual, commands)
        self.prediction = prediction
        trust = min(.95, self.confidence)
        if turn == 160:
            trust *= .65
        return prediction, stationary, trust

    def remember(self, own_w, flagplans):
        if self.prediction is None:
            return
        pred = self.prediction
        self.expected = {k: pred[k].copy() for k in 'FWS'}
        for p, w in enumerate(own_w):
            self.expected['W'][p] = max(0, pred['W'][p] - w)
            if w > pred['W'][p]:
                self.expected['F'][p] = self.expected['S'][p] = 0


def project(g, base, actual, commands):
    """Apply the model's legal commands, retaining separate departure pools."""
    pool = {k: actual[k].copy() for k in 'FWS'}
    rows = [c.split() for c in commands]
    for row in rows:
        if row[0] == 'SPAWN':
            p = base if len(row) == 3 else g.cell(int(row[3]),int(row[4]))
            pool[row[1]][p] += int(row[2])
    stationary = {k: pool[k].copy() for k in 'FWS'}
    arrivals = {k: [0]*g.n for k in 'FWS'}
    for row in rows:
        if row[0] not in ('MOVE','TELE'):
            continue
        origin = g.cell(int(row[1]),int(row[2])); kind = row[3]
        count = min(int(row[4]),pool[kind][origin])
        if row[0] == 'TELE':
            dest = g.cell(int(row[5]),int(row[6]));count=min(count,5)
        else:
            dest = next(q for d,q in g.neighbors[origin] if d == row[5])
        pool[kind][origin] -= count
        arrivals[kind][dest] += count
    result = {k:[a+b for a,b in zip(pool[k],arrivals[k])] for k in 'FWS'}
    return stationary,result


def respond(g, buildings, team, enemy, prediction, stationary, trust,
            plans, flagplans, projected, acquire, loss, garrisons, deadline):
    """Improve one-turn combat/capture while retaining strategic goal guidance.

    A low-confidence model changes nothing. At higher confidence a weighted
    stationary fallback still charges for attacks that might not actually move.
    """
    if trust < .78 or perf_counter() >= deadline:
        return 0
    flags=[0]*g.n
    for _,q,c,_ in flagplans:flags[q]+=c
    scenarios=((prediction,trust),(stationary,1-trust))

    def tile(p):
        value=0.
        w,f=projected[p],flags[p]
        for hostile,weight in scenarios:
            ew,ef=hostile['W'][p],hostile['F'][p]
            a=f if w>=ew else 0
            b=ef if ew>=w else 0
            reward=6*a-6*b
            if p in buildings:
                original=buildings[p]['owner'];owner=original
                if original==team and f and not a:owner='N'
                elif original==enemy and ef and not b:owner='N'
                if a and not b and owner!=team:
                    owner=team if owner=='N' else 'N'
                elif b and not a and owner!=enemy:
                    owner=enemy if owner=='N' else 'N'
                own_value=loss[p] if original==team else acquire[p]/(1.85 if original==enemy else 1)
                enemy_value=acquire[p]*.85/1.85 if original==enemy else .85*loss[p]
                if owner==team:reward+=own_value
                elif owner==enemy:reward-=enemy_value
            value+=weight*reward
        return value

    changes=0
    for sweep in range(2):
        for plan in sorted(plans,key=lambda p:(-p[2],-p[4],g.key(p[0]))):
            if perf_counter() >= deadline:return changes
            origin,old,count,goal,value,role=plan
            if not count or role=='guard':continue
            guide=lambda q: .5*value*count**.35/(2+g.dist[q][goal])
            choice=old;best_gain=0.
            for q in g.adj[origin]:
                if q==old:continue
                if projected[old]-count<garrisons.get(old,0):continue
                before=tile(old)+tile(q)
                projected[old]-=count;projected[q]+=count
                gain=tile(old)+tile(q)-before+guide(q)-guide(old)
                projected[q]-=count;projected[old]+=count
                if gain>best_gain+1e-8:best_gain,choice=gain,q
            if choice!=old:
                projected[old]-=count;projected[choice]+=count;plan[1]=choice;changes+=1
        for plan in flagplans:
            if perf_counter() >= deadline:return changes
            origin,old,count,goal=plan
            if not count:continue
            value=acquire.get(goal,2.)
            guide=lambda q: .7*value/(2+g.dist[q][goal])
            choice=old;best_gain=0.
            for q in g.adj[origin]:
                if q==old:continue
                before=tile(old)+tile(q)
                flags[old]-=count;flags[q]+=count
                gain=tile(old)+tile(q)-before+guide(q)-guide(old)
                flags[q]-=count;flags[old]+=count
                if gain>best_gain+1e-8:best_gain,choice=gain,q
            if choice!=old:
                flags[old]-=count;flags[choice]+=count;plan[1]=choice;changes+=1
    return changes
