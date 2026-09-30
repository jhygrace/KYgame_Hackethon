"""Nova v6: deadline defense, lead locking and concentrated fronts.

Derived from the user's Nova v5. Uses only supplied turn observations.
The opponent model remains confidence-gated and has no access to hidden scores.
"""
from collections import Counter, defaultdict
from time import perf_counter
from board import Board, assign
from endgame import OwnershipHistory, finish
from tactics import consolidate
from predictor import OpponentModel, respond

class Nova:
    def __init__(self,terrain,bases,team,buildings):
        self.team=team;self.enemy='K' if team=='Y' else 'Y'
        self.board=Board(terrain,bases,team);self.base=self.board.cell(*bases[team]);self.enemy_base=self.board.cell(*bases[self.enemy])
        self.known={};self.depots=set();self.memory={};self.tele_history={};self.stats={};self.history=OwnershipHistory()
        self.lock_until=-1
        self.opponent_model=OpponentModel(terrain,bases,team,buildings)

    def decide(self,turn,money,enemy_money,units,buildings):
        began=perf_counter();g=self.board;n=g.n;left=161-turn
        bd={g.cell(x,y):{'id':i,'kind':kind,'owner':owner,'score':score} for i,x,y,kind,owner,stage,score in buildings}
        ours={k:[0]*n for k in 'FWS'};enemy={k:[0]*n for k in 'FWS'}
        for t,k,x,y,c in units:(ours if t==self.team else enemy)[k][g.cell(x,y)]+=c
        for p,b in bd.items():
            if b['score']>=0:self.known[p]=self.known[g.mirror(p)]=b['score']
            if b['owner']==self.team and b['kind']=='DEPOT':self.depots.add(p)
        self.history.observe(turn,bd,self.team,self.enemy)
        forecast,stationary,confidence=self.opponent_model.forecast(turn,money,enemy_money,units,buildings,self.known)
        def score(p):
            if p in self.known:return self.known[p]
            if bd[p]['kind']=='PLAZA':return 3
            return 3 if 5<=g.xy(p)[0]<=9 else 1.5
        owned={t:Counter(b['kind'] for b in bd.values() if b['owner']==t) for t in (self.team,self.enemy)}
        sites=[self.base]+[p for p,b in bd.items() if b['owner']==self.team and b['kind']=='HOSPITAL']
        esites=[self.enemy_base]+[p for p,b in bd.items() if b['owner']==self.enemy and b['kind']=='HOSPITAL']
        sites.sort(key=g.key);esites.sort(key=g.key)
        wc=2 if owned[self.team]['ENG'] else 3;ewc=2 if owned[self.enemy]['ENG'] else 3
        eW=enemy['W'];oW=ours['W'];oF=ours['F'];eF=enemy['F']
        own_w_total=sum(oW);enemy_w_total=sum(eW)
        own_buildings=sum(b['owner']==self.team for b in bd.values())
        enemy_buildings=sum(b['owner']==self.enemy for b in bd.values())
        territory_lead=own_buildings-enemy_buildings
        # When territory is already won, stop opening new fronts.  The old bot
        # kept six flags and three advance lanes active even while its W army
        # was losing the concentration battle.  This uses public ownership only.
        raw_lock=((turn>=70 and territory_lead>=4) or
                  (turn>=100 and territory_lead>=2) or
                  (own_buildings>=10 and enemy_buildings<=5))
        if raw_lock:self.lock_until=max(self.lock_until,turn+10)
        # Hysteresis: do not reopen several attack fronts the instant a large
        # lead shrinks. Keep defending for a short window while still tied/ahead.
        lock_mode=raw_lock or (turn<=self.lock_until and territory_lead>=0)
        understrength=enemy_w_total>=max(12,int(1.22*max(1,own_w_total)))
        reach=[0]*n;flagreach=[0]*n;friendly=[0]*n
        for p in range(n):
            for q in g.adj[p]:
                reach[q]+=eW[p];flagreach[q]+=eF[p];friendly[q]+=oW[p]
        present_reach=reach.copy()
        existing_flagreach=flagreach.copy()
        spawn_zone={q for p in esites for q in g.adj[p]}
        for q in spawn_zone:
            reach[q]+=enemy_money//ewc
            if enemy_money>=5:flagreach[q]+=1
        estations=[p for p,b in bd.items() if b['owner']==self.enemy and b['kind']=='STATION']
        if len(estations)>1:
            for p in estations:
                amount=min(5,max(eW[q] for q in estations if q!=p))
                reach[p]+=amount;present_reach[p]+=amount
        def near(sources,p):return min(g.dist[q][p] for q in sources)
        enemies_f=[p for p in range(n) if eF[p]]
        own_eta={p:near(sites,p) for p in bd};enemy_eta={p:near(esites,p) for p in bd}
        ef_eta={p:min((g.dist[q][p] for q in enemies_f),default=99) for p in bd}
        # To attack a building with a newly made F, the opponent must spend 5
        # on that F: it cannot also spend the same money on W.
        capture_threat={p:present_reach[p]+(max(0,enemy_money-(0 if existing_flagreach[p] else 5))//ewc if p in spawn_zone else 0) for p in bd}
        urgent={p for p,b in bd.items() if b['owner']==self.team and flagreach[p]}
        targets=[p for p,b in bd.items() if b['owner']!=self.team]
        targets.sort(key=g.key)
        pointweight=3.3+2.7*max(0,(turn-100)/60)

        # Production distance is not capture ETA: existing flags may already be
        # at the objective. Local forces bound the likely useful holding period.
        flags={self.team:[p for p in range(n) if oF[p]],self.enemy:enemies_f}
        capture_eta={t:{p:min(min((g.dist[q][p] for q in flags[t]),default=99),
                              (own_eta if t==self.team else enemy_eta)[p]+1)
                        for p in bd} for t in (self.team,self.enemy)}
        local={t:{} for t in (self.team,self.enemy)}
        for t,troops in ((self.team,oW),(self.enemy,eW)):
            sources=[(q,c) for q,c in enumerate(troops) if c]
            for p in bd:
                local[t][p]=sum(c*(1-.18*g.dist[q][p]) for q,c in sources if g.dist[q][p]<=4)
        holding={p:max(.30,min(1.,.65+.025*(enemy_eta[p]-own_eta[p])+
                              .35*(local[self.team][p]-local[self.enemy][p])/
                              (8+local[self.team][p]+local[self.enemy][p]))) for p in bd}

        def effect(p,team,counts,acquiring):
            """Marginal production/resource benefit with/without this building."""
            b=bd[p];kind=b['kind'];own=team==self.team
            eta=capture_eta[team][p] if acquiring else 0
            horizon=max(0,min(30,left-eta-1))
            # Reinforcement travel asymmetry limits optimistic holding horizons.
            advantage=(enemy_eta[p]-own_eta[p])*(1 if own else -1)
            hold=holding[p] if own else max(.30,min(1.,1.30-holding[p]))
            horizon*=hold
            income=10+2*counts['HALL'];cost=2 if counts['ENG'] else 3
            if kind=='HALL':return .85*(2/cost)*horizon
            if kind=='ENG':
                crucial=counts['ENG']==(0 if acquiring else 1)
                return .85*(income/2-income/3)*horizon if crucial else 0.
            if kind=='LIBRARY':
                crucial=counts['LIBRARY']==(0 if acquiring else 1)
                return .45*min(10,len(targets))*min(1,horizon/16) if crucial else 0.
            if kind=='DEPOT':
                return .29*min(15,max(0,40-(10+2*counts['HALL']))) if acquiring and own and p not in self.depots and left>eta+1 else 0.
            if kind=='WATCH' and acquiring and horizon>3:
                unknown=set()
                x,y=g.xy(p)
                for q in bd:
                    xx,yy=g.xy(q)
                    if max(abs(x-xx),abs(y-yy))<=3 and q not in self.known and q not in (p,g.mirror(p)):
                        unknown.add(min(q,g.mirror(q)))
                return .35*len(unknown)*min(1,horizon/15)
            if kind=='STATION':
                network=(counts['STATION']==1 if acquiring else counts['STATION']==2)
                others=[q for q,z in bd.items() if q!=p and z['kind']=='STATION' and z['owner']==team]
                span=max((g.dist[p][q] for q in others),default=0)
                return (min(4,span*.35) if network else .6)*min(1,horizon/15)
            if kind=='HOSPITAL':
                current=sites if own else esites
                old=[q for q in current if acquiring or q!=p]
                goals=sorted(targets,key=lambda q:-score(q))[:6]
                if not goals:return 0.
                savings=sum(max(0,near(old,q)-g.dist[p][q]) for q in goals)/len(goals)
                # A hospital is not merely a travel shortcut: it is a new spawn
                # origin.  Reward defensive coverage of valuable held buildings
                # as well, which prevents remote territory from becoming hollow.
                held=[q for q,z in bd.items() if z['owner']==team and q!=p]
                coverage=0.
                if held:
                    coverage=sum(score(q)*max(0,min(6,near(old,q))-min(6,g.dist[p][q]))
                                 for q in held)/max(3,len(held))
                return (1.00*savings+.60*coverage)*min(1,horizon/12)
            return 0.
        acquire={};loss={}
        for p,b in bd.items():
            value=pointweight*score(p)+effect(p,self.team,owned[self.team],True)
            if b['owner']==self.enemy:value+=.85*pointweight*score(p)+.70*effect(p,self.enemy,owned[self.enemy],False)
            acquire[p]=value
            loss[p]=pointweight*score(p)+effect(p,self.team,owned[self.team],False)

        spawns=Counter();budget=money
        def recruit(site,kind,count):
            nonlocal budget
            cost=5 if kind=='F' else wc
            count=min(count,budget//cost)
            if count:
                budget-=cost*count;spawns[site,kind]+=count;ours[kind][site]+=count
            return count
        # Buy a minimal immediate guard before optional expansion flags. Only
        # count production that can actually reach the threatened tile now.
        for p in sorted(urgent,key=lambda p:(-loss[p],g.key(p))):
            # Existing enemy F is the most urgent case, not a reason to skip.
            # One more friendly W than hostile W reach is enough to kill/deny
            # an arriving flag under the observed combat rule.
            needed=max(0,present_reach[p]+1-friendly[p])
            close=[s for s in sites if g.dist[s][p]<=1]
            if needed and close:
                site=min(close,key=lambda s:(g.dist[s][p],g.key(s)))
                made=recruit(site,'W',needed)
                for q in g.adj[site]:friendly[q]+=made

        def timely_value(origin,p):
            # A new unit moves immediately. Distance 1 therefore captures on
            # the production turn, while distance 0 still needs this turn.
            arrival=max(1,g.dist[origin][p])
            if arrival>left:return 0.
            if bd[p]['owner']==self.enemy and arrival+1>left:
                # Neutralizing an empty enemy building is still a score gain
                # even when there is no following turn to finish the capture.
                return acquire[p]*.50
            return acquire[p]
        # Match existing flag tokens globally. A dummy column means rest safely.
        def match_flags():
            origins=[p for p in range(n) for _ in range(oF[p])]
            origins.sort(key=g.key)
            if not origins:return []
            costs=[]
            for origin in origins:
                row=[]
                for p in targets:
                    d=g.dist[origin][p];extra=int(bd[p]['owner']==self.enemy and not eF[p])
                    value=timely_value(origin,p)
                    if not value:row.append(10000.);continue
                    # Waiting for an escort has an opportunity cost. Recompute
                    # every turn, allowing a lost center to become viable again.
                    deficit=max(0,local[self.enemy][p]-local[self.team][p])
                    wait=min(8,deficit/max(1,(10+2*owned[self.team]['HALL'])/wc))
                    if left<=15:wait=min(wait,max(0,left-max(1,d)-extra))
                    util=value/(2+d+extra+wait)**.85
                    util-=.045*max(0,eW[p]-friendly[p])
                    if p in self.memory.get(origin,()):util*=1.15
                    row.append(-util)
                costs.append(row+[0.]*len(origins))
            assignment=assign(costs)
            return [(origins[i],targets[col]) for i,col in enumerate(assignment) if 0<=col<len(targets) and costs[i][col]<0]
        # Keep only as many capture fronts as the W army can plausibly escort.
        # v5 reached six flags on turns 3-4 and permanently fragmented its army.
        if lock_mode:
            max_flags=3 if understrength or territory_lead>=6 else 4
        else:
            escort_cap=4+min(2,own_w_total//36)
            max_flags=min(6,max(3,min(len(targets),escort_cap)))
        tasks=match_flags()
        if len(tasks)>max_flags:
            tasks=sorted(tasks,key=lambda og:(-(acquire[og[1]]/(2+g.dist[og[0]][og[1]])),g.key(og[0]),g.key(og[1])))[:max_flags]
        reserved={p for _,p in tasks}
        for _ in range(2):
            if budget<5 or sum(oF)>=max_flags:break
            options=[]
            for site in sites:
                for p in targets:
                    if p in reserved:continue
                    d=g.dist[site][p]
                    if d>min(11,left):continue
                    if reach[site]>oW[site]+max(0,budget-5)//wc:continue
                    deficit=max(0,local[self.enemy][p]-local[self.team][p])
                    wait=min(8,deficit/max(1,(10+2*owned[self.team]['HALL'])/wc))
                    if left<=15:wait=min(wait,max(0,left-max(1,d)))
                    util=timely_value(site,p)/(3+d+wait)**.9
                    options.append((util,-d,-g.key(p),-g.key(site),site,p))
            if not options:break
            utility,_,_,_,site,p=max(options)
            if utility<.55:break
            if recruit(site,'F',1):tasks.append((site,p));reserved.add(p)
        # Reserve only capture costs not covered by the next guaranteed income.
        potential={q for p in range(n) if oF[p] for q in g.adj[p] if q in bd and bd[q]['owner']!=self.team}
        cap_need=sum(max(1,(4 if bd[p]['kind']=='PLAZA' else 2)-bool(owned[self.team]['LIBRARY'])) for p in potential)
        capture_reserve=max(0,cap_need-10)

        # Strategic jobs describe useful future positions, not per-unit scripts.
        jobs={}
        def add(p,need,value,role):
            need=max(1,int(need))
            if p not in jobs:jobs[p]=[need,value,role]
            else:
                jobs[p][0]=max(jobs[p][0],need)
                if value>jobs[p][1]:jobs[p][1:]=[value,role]
        for p,b in bd.items():
            if b['owner']==self.team and (ef_eta[p]<=5 or p in urgent):
                immediate=p in urgent
                demand=max(1,capture_threat[p]+1) if immediate else max(1,eW[p]+1)
                eta=min(ef_eta[p],1) if immediate else ef_eta[p]
                add(p,demand,loss[p]*(2.4 if immediate else 1.5)/(1+.35*max(0,eta-1)),'defend')
        for origin,goal in tasks:
            choices=[q for q in g.adj[origin] if g.dist[q][goal]<g.dist[origin][goal]] or [origin]
            nextp=min(choices,key=lambda q:(max(0,reach[q]-friendly[q]),g.dist[q][goal],g.key(q)))
            if reach[nextp] or (nextp in bd and flagreach[nextp]):
                add(nextp,reach[nextp]+int(bool(flagreach[nextp])),2.4*acquire[goal]/(1+.15*g.dist[origin][goal]),'escort')
            add(goal,max(2,reach[goal]+1 if g.dist[origin][goal]<=2 else 3),acquire[goal]*1.25,'capture')
        for p in enemies_f:
            value=8+min(eF[p],3)*3
            if left<=14:value*=1.35
            if p in bd:value+=loss[p] if bd[p]['owner']==self.team else acquire[p]
            add(p,reach[p]+1,value,'intercept')
        # In a lead, turn the highest-value holdings into anchors instead of
        # continuously creating fresh attack lanes.  These are normal defend
        # jobs; hard local reservation is added below when troops are available.
        hold_targets=[]
        if lock_mode:
            held=[p for p,b in bd.items() if b['owner']==self.team]
            held.sort(key=lambda p:(-(loss[p]*(1.25 if bd[p]['kind'] in ('HOSPITAL','ENG','STATION','PLAZA') else 1)),
                                    ef_eta[p],g.key(p)))
            hold_targets=held[:min(5,max(3,own_buildings//2))]
            for p in hold_targets:
                need=max(2,min(4,reach[p]+1))
                add(p,need,loss[p]*1.65,'defend')
        # Unassigned troops use one main front while locking a lead or when
        # outnumbered; only a clear W advantage permits three attack lanes.
        advance_lanes=1 if (lock_mode or understrength) else (3 if own_w_total>1.35*max(1,enemy_w_total)+12 else 2)
        for p in sorted(targets,key=lambda q:-(acquire[q]/(3+own_eta[q])))[:advance_lanes]:
            add(p,max(4,eW[p]+1),acquire[p]*(1.10 if advance_lanes==1 else 1.0),'advance')
        if not jobs:add(self.base,1,1,'reserve')
        joblist=sorted(jobs.items(),key=lambda item:(-item[1][1],g.key(item[0])))
        # Spend all affordable production, including hospital sites, after F choices.
        recruited=Counter()
        while budget-capture_reserve>=wc:
            candidates=[]
            for p,(need,value,role) in joblist:
                for site in sites:
                    eta=g.dist[site][p]
                    if role=='defend':
                        defend_deadline=1 if p in urgent else max(1,min(5,ef_eta.get(p,5)))
                        if eta>defend_deadline:continue
                    benefit=value/((2+eta)*(1+recruited[p]/max(2,need)))
                    candidates.append((benefit,-g.key(site),-g.key(p),site,p))
            if not candidates:break
            _,_,_,site,p=max(candidates);recruit(site,'W',1);recruited[p]+=1

        # Allocate integer W demand; protect immediately threatened buildings first.
        free=oW.copy();alloc=[];garrisons={}
        # A concrete guard cannot be reassigned by the movement optimizer.
        # Use existing hostile W here, rather than requiring all hypothetical
        # enemy production at every spawn at the same time.
        for goal in sorted(urgent,key=lambda p:(-loss[p],g.key(p))):
            need=present_reach[goal]+1
            sources=sorted((q for q in g.adj[goal] if free[q]),key=lambda q:(q!=goal,-free[q],g.key(q)))
            if sum(free[q] for q in sources)<need:continue
            garrisons[goal]=need
            for origin in sources:
                take=min(need,free[origin])
                if take:alloc.append([origin,goal,take,loss[goal]*2.4,'guard']);free[origin]-=take;need-=take
                if not need:break
        # Lead-lock anchors: keep a small local floor at valuable holdings.
        # Do this only when the floor already exists nearby; never drag distant
        # troops away from an emergency to manufacture a static garrison.
        if lock_mode:
            for goal in hold_targets:
                if goal in garrisons:continue
                floor=max(1,min(3,1+int(loss[goal]>=pointweight*3)))
                sources=sorted((q for q in g.adj[goal] if free[q]),key=lambda q:(q!=goal,-free[q],g.key(q)))
                if sum(free[q] for q in sources)<floor:continue
                garrisons[goal]=floor;need=floor
                for origin in sources:
                    take=min(need,free[origin])
                    if take:alloc.append([origin,goal,take,loss[goal]*1.7,'guard']);free[origin]-=take;need-=take
                    if not need:break
        ordered=sorted(joblist,key=lambda item:(-item[1][1]/(1+.18*min((g.dist[q][item[0]] for q in range(n) if free[q]),default=30)),g.key(item[0])))
        for goal,(need,value,role) in ordered:
            need=max(0,need-garrisons.get(goal,0))
            pool=(q for q in range(n) if free[q])
            if role=='defend':
                # A defender arriving after the enemy F is not defense.  Urgent
                # threats have a one-move deadline; future threats use F ETA.
                deadline=1 if goal in urgent else max(1,min(5,ef_eta.get(goal,5)))
                pool=(q for q in pool if g.dist[q][goal]<=deadline)
            sources=sorted(pool,key=lambda q:(g.dist[q][goal],-free[q],g.key(q)))
            for origin in sources:
                take=min(need,free[origin])
                if take:alloc.append([origin,goal,take,value,role]);free[origin]-=take;need-=take
                if not need:break
        for origin,count in enumerate(free):
            if not count:continue
            goal,(need,value,role)=max(joblist,key=lambda item:(item[1][1]/(2+g.dist[origin][item[0]]),-g.key(item[0])))
            alloc.append([origin,goal,count,value,role])
        # Joint coordinate descent over one-turn W destinations. Arrivals are kept
        # distinct from movable troops, which also makes TELE composition exact.
        plans=[];projected=oW.copy()
        for origin,goal,count,value,role in alloc:
            q=goal if role=='guard' else min(g.adj[origin],key=lambda q:(g.dist[q][goal],max(0,reach[q]-count-oW[q]),g.key(q)))
            plans.append([origin,q,count,goal,value,role]);projected[origin]-=count;projected[q]+=count
        for sweep in range(3):
            for plan in sorted(plans,key=lambda p:(-p[2],-p[4],g.key(p[0]))):
                if plan[5]=='guard':continue
                origin,old,count,goal,value,role=plan;projected[old]-=count
                def movement_cost(q):
                    shortfall=max(0,reach[q]-projected[q]-count)
                    # Full hostile reach is a safety envelope, not an enemy forecast.
                    danger=2.6*shortfall/max(1,count)**.35
                    home_exposure=0.
                    if origin in bd and bd[origin]['owner']==self.team and flagreach[origin] and q!=origin:
                        if projected[origin]<=reach[origin]:home_exposure=loss[origin]*.12
                    return g.dist[q][goal]+danger+home_exposure+.05*(q==origin and origin!=goal)
                q=min(g.adj[origin],key=lambda q:(movement_cost(q),g.dist[q][goal],g.key(q)))
                plan[1]=q;projected[q]+=count
            if perf_counter()-began>.13:break
        regrouped=consolidate(g,plans,projected,reach,flagreach,garrisons,began+.16)
        # A* evaluates candidate first steps using the final ordinary W deployment.
        flagplans=[];used=Counter();memory=defaultdict(set);cache={}
        penalty=[20*max(0,reach[p]-projected[p]) for p in range(n)]
        for origin,goal in tasks:
            safe=[q for q in g.adj[origin] if reach[q]<=projected[q]]
            options=safe or g.adj[origin]
            for q in options:
                if (q,goal) not in cache:cache[q,goal]=g.astar_cost(q,goal,penalty)
            q=min(options,key=lambda q:(max(0,reach[q]-projected[q]),cache[q,goal]+(0 if q==origin else 1+penalty[q]),int(q==origin and q!=goal),g.key(q)))
            flagplans.append([origin,q,1,goal]);used[origin]+=1;memory[q].add(goal)
        for origin,count in enumerate(oF):
            count-=used[origin]
            if count<=0:continue
            q=min(g.adj[origin],key=lambda q:(max(0,reach[q]-projected[q]),int(q in bd and bd[q]['owner']==self.team and reach[q]>projected[q]),near(sites,q),g.key(q)))
            flagplans.append([origin,q,count,origin])
        # TELE replaces an already-planned departure; it never steals a reserved
        # garrison. Validate BOTH affected destinations after all ordinary moves.
        final_flags=[0]*n
        for p,q,c,goal in flagplans:final_flags[q]+=c
        stations=[p for p,b in bd.items() if b['owner']==self.team and b['kind']=='STATION']
        teleport=None;tcandidates=[]
        if len(stations)>=2:
            for idx,plan in enumerate(plans):
                src,old,count,goal,value,role=plan
                if role=='guard':continue
                if src not in stations or old==src:continue
                for dst in stations:
                    if dst==src:continue
                    gain=g.dist[old][goal]-g.dist[dst][goal]
                    amount=min(5,count)
                    if gain<3 or amount<2:continue
                    if turn-self.tele_history.get((dst,src),-100)<5:continue
                    if projected[dst]+amount<reach[dst]:continue
                    remaining=projected[old]-amount
                    if remaining<reach[old] and (final_flags[old] or (old in bd and bd[old]['owner']==self.team and flagreach[old])):continue
                    tcandidates.append((gain*amount*value/(3+g.dist[dst][goal]),-g.key(dst),idx,src,dst,amount))
            if tcandidates:
                _,_,idx,src,dst,amount=max(tcandidates);old=plans[idx][1];plans[idx][2]-=amount
                projected[old]-=amount;projected[dst]+=amount;teleport=(src,'W',amount,dst)
                self.tele_history[src,dst]=turn
        prediction_changes=respond(g,bd,self.team,self.enemy,forecast,stationary,confidence,
                                   plans,flagplans,projected,acquire,loss,garrisons,began+.16)
        terminal_stats={};terminal_priority=None
        if left==1:
            terminal_priority,terminal_stats=finish(
                g,bd,self.team,self.enemy,ours,enemy,enemy_money,esites,ewc,
                budget,plans,flagplans,projected,teleport,score,
                self.history.interval(bd,self.known,g),began+.20)
        commands=[]
        for (site,kind),count in sorted(spawns.items(),key=lambda x:(g.key(x[0][0]),x[0][1])):
            text=f'SPAWN {kind} {count}'
            if site!=self.base:text+=' %d %d'%g.xy(site)
            commands.append(text)
        if teleport:
            src,kind,c,dst=teleport;x,y=g.xy(src);xx,yy=g.xy(dst);commands.append(f'TELE {x} {y} {kind} {c} {xx} {yy}')
        moves=Counter()
        for origin,q,c,*_ in plans:
            if c and q!=origin:moves[origin,'W',g.direction(origin,q)]+=c
        for origin,q,c,*_ in flagplans:
            if c and q!=origin:moves[origin,'F',g.direction(origin,q)]+=c
        for (origin,kind,direction),c in sorted(moves.items(),key=lambda x:(x[0][1],g.key(x[0][0]),x[0][2])):
            x,y=g.xy(origin);commands.append(f'MOVE {x} {y} {kind} {c} {direction}')
        priorities=sorted({q for _,q,c,_ in flagplans if c and q in bd and bd[q]['owner']!=self.team},key=lambda p:(-acquire[p],g.key(p)))
        if terminal_priority is not None:
            priorities=[p for p in terminal_priority if any(q==p and c for _,q,c,_ in flagplans)]
        if priorities:commands.append('PRIORITY '+' '.join('%d %d'%g.xy(p) for p in priorities))
        memory=defaultdict(set)
        for _,q,c,goal in flagplans:
            if c:memory[q].add(goal)
        self.opponent_model.remember(projected,flagplans)
        self.memory=dict(memory);self.stats={'turn':turn,'ms':1000*(perf_counter()-began),'flags':sum(oF),'tele':int(teleport is not None),'regrouped':regrouped,'guards':sum(garrisons.values()),'lock_mode':int(lock_mode),'territory_lead':territory_lead,'prediction_confidence':confidence,'prediction_changes':prediction_changes}
        self.stats.update(terminal_stats)
        return commands
