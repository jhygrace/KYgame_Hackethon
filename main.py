"""Nova v6: Campus Conquest entrypoint (Python standard library only)."""
import sys
from bot import Nova

def block():
    out=[]
    for line in sys.stdin:
        line=line.strip()
        if line=='END':return out
        if line:out.append(line)
    return None

def main():
    raw=block()
    if raw is None:return
    _,w,h=raw[0].split();w,h=int(w),int(h);team=raw[1].split()[1]
    terrain=[raw[2+i].split()[1] for i in range(h)];i=2+h
    count=int(raw[i].split()[1]);i+=1;buildings=[]
    for line in raw[i:i+count]:
        ident,x,y,kind=line.split();buildings.append((int(ident),int(x),int(y),kind))
    i+=count;bases={}
    for line in raw[i:i+2]:
        _,t,x,y=line.split();bases[t]=(int(x),int(y))
    bot=Nova(terrain,bases,team,buildings);errors=0
    while True:
        raw=block()
        if raw is None:return
        turn=int(raw[0].split()[1]);_,own,enemy=raw[1].split();n=int(raw[2].split()[1]);i=3;units=[]
        for line in raw[i:i+n]:
            t,k,x,y,c=line.split();units.append((t,k,int(x),int(y),int(c)))
        i+=n;n=int(raw[i].split()[1]);i+=1;buildings=[]
        for line in raw[i:i+n]:
            ident,x,y,k,o,stage,score=line.split();buildings.append((int(ident),int(x),int(y),k,o,int(stage),int(score)))
        try:
            commands=bot.decide(turn,int(own),int(enemy),units,buildings)
        except Exception as exc:
            if errors<4:
                print(f'Nova error: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
                errors+=1
            commands=[]
        sys.stdout.write('\n'.join(commands+['END'])+'\n');sys.stdout.flush()

if __name__=='__main__':main()
