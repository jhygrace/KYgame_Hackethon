"""Nova v4: Floyd-Warshall geometry, A* risk paths, Hungarian assignment."""
from heapq import heappop, heappush

INF = 10**6

class Board:
    def __init__(self, terrain, bases, team):
        self.h = len(terrain)
        self.w = len(terrain[0])
        self.n = self.h * self.w
        self.flip = bases[team][0] > (self.w-1)/2
        self.order = [('U',0,-1),('L',-1,0),('D',0,1),('R',1,0)]
        if self.flip:
            self.order = [('D',0,1),('R',1,0),('U',0,-1),('L',-1,0)]
        self.neighbors = [[] for _ in range(self.n)]
        self.adj = [[] for _ in range(self.n)]
        for p in range(self.n):
            x,y = self.xy(p)
            if terrain[y][x] == '#':
                continue
            for name,dx,dy in self.order:
                xx,yy=x+dx,y+dy
                if 0<=xx<self.w and 0<=yy<self.h and terrain[yy][xx]!='#':
                    self.neighbors[p].append((name, self.cell(xx,yy)))
            self.adj[p]=[p]+[q for _,q in self.neighbors[p]]
        # Static terrain only: compute once per game, never once per turn.
        # Keep full cell indexing for the policy; exclude walls from FW loops.
        self.dist=[[INF]*self.n for _ in range(self.n)]
        for p in range(self.n):
            self.dist[p][p]=0
            for _,q in self.neighbors[p]:self.dist[p][q]=1
        walkable=[p for p in range(self.n) if self.adj[p]]
        for k in walkable:
            dk=self.dist[k]
            destinations=[(j,dk[j]) for j in walkable if dk[j]<INF]
            for i in walkable:
                di=self.dist[i];via=di[k]
                if via>=INF:continue
                for j,kj in destinations:
                    candidate=via+kj
                    if candidate<di[j]:di[j]=candidate

    def cell(self,x,y):return y*self.w+x
    def xy(self,p):return p%self.w,p//self.w
    def key(self,p):return self.n-1-p if self.flip else p
    def mirror(self,p):return self.n-1-p
    def direction(self,p,q):
        return next((d for d,x in self.neighbors[p] if x==q),None)
    def astar_cost(self,start,goal,penalty):
        """Minimum sum of 1 + penalty[destination] along a path.

        Static Floyd-Warshall distance is an admissible, consistent heuristic:
        penalties are nonnegative and each move costs at least one. The caller
        compares stay/adjacent candidates and emits only the chosen first step.
        This preserves v3's risk objective and deterministic first-step ties.
        """
        if start==goal:return 0.0
        if self.dist[start][goal]>=INF:return float('inf')
        distance=[float('inf')]*self.n;distance[start]=0.0
        heap=[(self.dist[start][goal],0.0,self.key(start),start)]
        while heap:
            _,value,_,p=heappop(heap)
            if value!=distance[p]:continue
            if p==goal:return value
            for _,q in self.neighbors[p]:
                v=value+1.0+penalty[q]
                if v<distance[q]:
                    distance[q]=v
                    heappush(heap,(v+self.dist[q][goal],v,self.key(q),q))
        return float('inf')


def assign(cost):
    """Hungarian algorithm: n rows <= m columns, one distinct column per row."""
    if not cost:return []
    n,m=len(cost),len(cost[0]);u=[0.]*(n+1);v=[0.]*(m+1);p=[0]*(m+1);way=[0]*(m+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;best=[float('inf')]*(m+1);used=[False]*(m+1)
        while True:
            used[j0]=True;i0=p[j0];delta=float('inf');j1=0
            for j in range(1,m+1):
                if not used[j]:
                    cur=cost[i0-1][j-1]-u[i0]-v[j]
                    if cur<best[j]:best[j]=cur;way[j]=j0
                    if best[j]<delta:delta=best[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:best[j]-=delta
            j0=j1
            if not p[j0]:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if not j0:break
    result=[-1]*n
    for j in range(1,m+1):
        if p[j]:result[p[j]-1]=j-1
    return result
