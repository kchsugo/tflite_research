# -*- coding: utf-8 -*-
# 통짜(multitask) vs split(jackson) 비교 + 50-task 통짜 결과 시각화 → docs/figures/*.svg
import os, sys, math
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
TXT="#222"; GRID="#ddd"; CPU_COL="#4e79a7"; GPU_COL="#e15759"; GREEN="#59a14f"; RED="#e15759"; ORANGE="#f28e2b"
def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def rect(x,y,w,h,fill,stroke="none"): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.1f}" height="{max(0,h):.1f}" fill="{fill}" stroke="{stroke}"/>'
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s):
    open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

# ---- Fig 6: 통짜 vs split head-to-head ----
def fig_compare():
    wls=["each2\n(10)","each3\n(15)","mob6\n(6)","bise4\n(4)","mix10\n(10)"]
    whole=[1.04,1.10,0.43,1.01,1.04]; split=[1.12,1.04,0.90,0.95,1.01]
    W,H=760,400; b=[text(20,28,"[Fig 6] 통짜(multitask) vs split(jackson) speedup 비교 (vs ALL-GPU)",15,weight="bold")]
    x0=70;y0=55;ph=250;grp=130;bw=45;mxv=1.2;mnv=0.4
    def sy(v): return y0+ph*(mxv-v)/(mxv-mnv)
    for gv in [0.4,0.6,0.8,1.0,1.2]:
        b.append(line(x0,sy(gv),x0+len(wls)*grp,sy(gv),GRID if gv!=1.0 else "#000",1 if gv!=1.0 else 1.5,"" if gv!=1.0 else "4"))
        b.append(text(x0-8,sy(gv)+4,f"{gv:.1f}",10,"end",fill="#999"))
    x=x0+20
    for i,w in enumerate(wls):
        for val,col,dx,lab in ((whole[i],ORANGE,0,"통짜"),(split[i],CPU_COL,bw+8,"split")):
            top=sy(max(val,1.0)); bot=sy(min(val,1.0))
            b.append(rect(x+dx,top,bw,bot-top,col if val>=1 else "#c0392b" if col==ORANGE else "#8e44ad"))
            b.append(text(x+dx+bw/2,(top-5 if val>=1 else bot+14),f"{val:.2f}",10,"middle",weight="bold"))
        for j,ln in enumerate(w.split("\n")): b.append(text(x+bw+4,y0+ph+18+j*13,ln,10,"middle",fill="#444"))
        x+=grp
    b.append(rect(x0,H-40,14,12,ORANGE)); b.append(text(x0+20,H-30,"통짜(whole-task routing)",11))
    b.append(rect(x0+200,H-40,14,12,CPU_COL)); b.append(text(x0+220,H-30,"split(jackson pipeline)",11))
    b.append(text(x0,H-10,"통짜는 mob6에서 0.43x 참사(6개 다 CPU로). split이 더 안정적이나 ~15개까지만 스케일.",11,fill="#a00"))
    save("fig6_compare.svg",svg(W,H,"".join(b)))

# ---- Fig 7: 50-task 통짜 배분 (모델별 GPU/CPU) ----
def fig_mt50_assign():
    # (모델, GPU개수, CPU개수)
    data=[("mobilenet",0,12),("yolov4",5,6),("lanenet",10,0),("BiseNet",8,0),("WideResNet",9,0)]
    W,H=720,300; b=[text(20,28,"[Fig 7] 50-task 통짜 배분 (모델별 GPU/CPU 개수)  GPU=32, CPU=18",15,weight="bold")]
    x0=120;y0=55;bw=90;gap=30;maxc=12;ph=180
    def sh(n): return ph*n/maxc
    x=x0
    for nm,g,c in data:
        gy=y0+ph-sh(g); b.append(rect(x,gy,bw,sh(g),GPU_COL))
        cy=gy-sh(c); b.append(rect(x,cy,bw,sh(c),CPU_COL))
        if g: b.append(text(x+bw/2,gy+sh(g)/2+4,f"G{g}",11,"middle",fill="#fff",weight="bold"))
        if c: b.append(text(x+bw/2,cy+sh(c)/2+4,f"C{c}",11,"middle",fill="#fff",weight="bold"))
        b.append(text(x+bw/2,y0+ph+16,nm,10,"middle")); x+=bw+gap
    b.append(rect(x0,H-38,14,12,GPU_COL)); b.append(text(x0+20,H-28,"GPU 배정",11))
    b.append(rect(x0+120,H-38,14,12,CPU_COL)); b.append(text(x0+140,H-28,"CPU 배정",11))
    b.append(text(x0,H-8,"가벼운 mobilenet(12개)·yolov4(6개)만 CPU로, 무거운 lan/bise/wide는 전부 GPU.",11,fill="#555"))
    save("fig7_mt50_assign.svg",svg(W,H,"".join(b)))

# ---- Fig 8: 50-task 통짜 makespan (로그축) ----
def fig_mt50_ms():
    data=[("ALL-GPU\n308.5ms",308.53,GPU_COL),("SCHEDULED\n322.9ms",322.90,ORANGE),("ALL-CPU\n2780ms",2780.22,CPU_COL)]
    W,H=620,340; b=[text(20,28,"[Fig 8] 50-task 통짜 makespan 실측 (로그축)  speedup 0.96x (손해)",15,weight="bold")]
    x0=110;y0=55;ph=200;bw=110;gap=50
    lo,hi=math.log10(100),math.log10(3000)
    def sy(v): return y0+ph*(hi-math.log10(v))/(hi-lo)
    for gv in [100,300,1000,3000]:
        b.append(line(x0,sy(gv),x0+len(data)*(bw+gap),sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv}ms",10,"end",fill="#999"))
    x=x0+gap/2
    for lab,v,col in data:
        top=sy(v); b.append(rect(x,top,bw,y0+ph-top,col)); b.append(text(x+bw/2,top-6,f"{v:.0f}",12,"middle",weight="bold"))
        for i,ln in enumerate(lab.split("\n")): b.append(text(x+bw/2,y0+ph+18+i*14,ln,11,"middle",fill="#444"))
        x+=bw+gap
    b.append(text(x0,H-14,"SCHEDULED(322)가 ALL-GPU(308)보다 살짝 느림 — 가벼운걸 CPU로 뺀 오버헤드가 이득 초과.",11,fill="#a00"))
    save("fig8_mt50_ms.svg",svg(W,H,"".join(b)))

fig_compare(); fig_mt50_assign(); fig_mt50_ms(); print("DONE")
