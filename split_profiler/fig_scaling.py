# -*- coding: utf-8 -*-
# N(task수) vs makespan / speedup 추이 → docs/figures/*.svg
import os,sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
TXT="#222"; GRID="#ddd"; GPU_COL="#e15759"; ORANGE="#f28e2b"; GREEN="#59a14f"
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def circ(x,y,r,fill): return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'
def rect(x,y,w,h,fill,op=1): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" opacity="{op}"/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

N=[50,70,90,100,120]
allgpu=[532.85,828.38,1052.96,1128.36,1329.26]
sched=[497.28,811.02,1052.97,1217.58,1549.34]
sp=[1.072,1.021,1.000,0.927,0.858]

# ---- Fig 9: makespan vs N ----
def fig_ms():
    W,H=720,420; b=[text(20,28,"[Fig 9] task 수(N)별 makespan — ALL-GPU vs SCHEDULED(통짜 배분)",15,weight="bold")]
    x0=70;y0=55;pw=600;ph=290;mx=1600
    def sx(n): return x0+pw*(n-40)/(130-40)
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,1601,400):
        b.append(line(x0,sy(gv),x0+pw,sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv}",10,"end",fill="#999"))
    for n in N:
        b.append(line(sx(n),y0,sx(n),y0+ph,GRID)); b.append(text(sx(n),y0+ph+18,f"{n}",11,"middle"))
    b.append(text(x0+pw/2,y0+ph+38,"task 수 N",12,"middle"))
    def poly(vals,col):
        pts=" ".join(f"{sx(N[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(N)))
        s=f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>'
        for i in range(len(N)): s+=circ(sx(N[i]),sy(vals[i]),4,col)
        return s
    b.append(poly(allgpu,GPU_COL)); b.append(poly(sched,ORANGE))
    b.append(rect(x0+pw-160,y0+5,14,10,GPU_COL)); b.append(text(x0+pw-142,y0+14,"ALL-GPU",11))
    b.append(rect(x0+pw-160,y0+22,14,10,ORANGE)); b.append(text(x0+pw-142,y0+31,"SCHEDULED",11))
    b.append(text(x0,H-12,"N 작을 땐 SCHEDULED가 아래(이득), N 커지면 위로 교차(손해). ~90에서 만남.",11,fill="#a00"))
    save("fig9_scaling_ms.svg",svg(W,H,"".join(b)))

# ---- Fig 10: speedup vs N ----
def fig_sp():
    W,H=720,400; b=[text(20,28,"[Fig 10] task 수(N)별 speedup (vs ALL-GPU) — 많아질수록 악화",15,weight="bold")]
    x0=70;y0=55;pw=600;ph=270;mxv=1.15;mnv=0.80
    def sx(n): return x0+pw*(n-40)/(130-40)
    def sy(v): return y0+ph*(mxv-v)/(mxv-mnv)
    b.append(rect(x0,y0,pw,sy(1.0)-y0,GREEN,0.08)); b.append(rect(x0,sy(1.0),pw,y0+ph-sy(1.0),GPU_COL,0.08))
    for gv in [0.85,0.9,0.95,1.0,1.05,1.10]:
        b.append(line(x0,sy(gv),x0+pw,sy(gv),"#000" if gv==1.0 else GRID,1.5 if gv==1.0 else 1,"4" if gv==1.0 else ""))
        b.append(text(x0-8,sy(gv)+4,f"{gv:.2f}",10,"end",fill="#999"))
    for n in N: b.append(line(sx(n),y0,sx(n),y0+ph,GRID)); b.append(text(sx(n),y0+ph+18,f"{n}",11,"middle"))
    b.append(text(x0+pw/2,y0+ph+38,"task 수 N",12,"middle"))
    pts=" ".join(f"{sx(N[i]):.1f},{sy(sp[i]):.1f}" for i in range(len(N)))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="3"/>')
    for i in range(len(N)):
        col=GREEN if sp[i]>=1 else GPU_COL
        b.append(circ(sx(N[i]),sy(sp[i]),5,col)); b.append(text(sx(N[i]),sy(sp[i])-10,f"{sp[i]:.2f}x",11,"middle",weight="bold",fill=col))
    b.append(text(x0+8,sy(1.07),"이득",11,fill=GREEN)); b.append(text(x0+pw-60,sy(0.87),"손해",11,fill=GPU_COL))
    b.append(text(x0,H-12,"N=50 최고 1.07x → N 커질수록 하락 → N=120 0.86x. 많을수록 CPU offload↑ → 경합↑.",11,fill="#a00"))
    save("fig10_scaling_speedup.svg",svg(W,H,"".join(b)))

fig_ms(); fig_sp(); print("DONE")
