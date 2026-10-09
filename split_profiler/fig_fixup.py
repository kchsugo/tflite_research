# -*- coding: utf-8 -*-
# fig11(문제/수정 분리) + fig13(makespan 수정후, fig9스타일) + 모든 SVG에 XML선언 추가
import os,sys,glob
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures")
TXT="#222"; GRID="#ddd"; GPU_COL="#e15759"; CPU_COL="#4e79a7"; GREEN="#59a14f"; ORANGE="#f28e2b"; BLACK="#000"
XMLDECL='<?xml version="1.0" encoding="UTF-8"?>\n'
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def circ(x,y,r,fill): return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'
def rect(x,y,w,h,fill,op=1): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(0,h):.1f}" fill="{fill}" opacity="{op}"/>'
def svg(w,h,b): return XMLDECL+f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

N=[50,70,90,100,120]

def sides(title,gpu_side,cpu_side,all_gpu,note,fname):
    W,H=780,440; b=[text(20,28,title,15,weight="bold")]
    x0=70;y0=55;pw=640;ph=300;mx=1600;grp=pw/len(N);bw=42
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,1601,400):
        b.append(line(x0,sy(gv),x0+pw,sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv}",10,"end",fill="#999"))
    for i,n in enumerate(N):
        cx=x0+grp*i+grp/2
        b.append(rect(cx-bw-3,sy(gpu_side[i]),bw,y0+ph-sy(gpu_side[i]),GPU_COL))
        b.append(rect(cx+3,sy(cpu_side[i]),bw,y0+ph-sy(cpu_side[i]),CPU_COL))
        b.append(text(cx-bw/2-3,sy(gpu_side[i])-4,f"{gpu_side[i]:.0f}",9,"middle",fill=GPU_COL))
        b.append(text(cx+bw/2+3,sy(cpu_side[i])-4,f"{cpu_side[i]:.0f}",9,"middle",fill=CPU_COL))
        b.append(line(cx-bw-6,sy(all_gpu[i]),cx+bw+6,sy(all_gpu[i]),BLACK,2))
        b.append(text(cx,y0+ph+18,f"N={n}",11,"middle"))
        bott="CPU측 병목" if cpu_side[i]>gpu_side[i] else "GPU측 병목"
        b.append(text(cx,y0+ph+34,bott,9,"middle",fill=CPU_COL if cpu_side[i]>gpu_side[i] else GPU_COL))
    b.append(rect(x0,H-40,14,12,GPU_COL)); b.append(text(x0+20,H-30,"GPU측",11))
    b.append(rect(x0+110,H-40,14,12,CPU_COL)); b.append(text(x0+130,H-30,"CPU측",11))
    b.append(line(x0+220,H-34,x0+240,H-34,BLACK,2)); b.append(text(x0+246,H-30,"ALL-GPU 기준",11))
    b.append(text(x0,H-12,note,11,fill="#a00"))
    save(fname,svg(W,H,"".join(b)))

# fig11 (문제/수정전): CPU측이 GPU측을 넘음
sides("[Fig 11] (수정 전) N별 GPU측 vs CPU측 — CPU측이 GPU측을 넘어 병목",
      [480.55,678.24,883.90,1046.81,1211.32],[446.59,804.97,1095.77,1103.87,1476.14],
      [614.00,857.87,1019.47,1219.88,1356.11],
      "N>=70에서 CPU측(파랑)이 GPU측(빨강) 위 → CPU 과부하가 병목 → 손해.","fig11_sides.svg")

# fig11b (수정후): CPU측이 항상 GPU측 아래
sides("[Fig 11b] (수정 후) N별 GPU측 vs CPU측 — CPU측이 항상 GPU측 아래",
      [425.55,747.34,935.27,1027.24,1189.73],[263.90,423.21,784.98,824.73,800.71],
      [576.43,747.72,964.40,1112.99,1391.61],
      "CPU측(파랑)이 GPU측(빨강)보다 항상 낮음 → CPU 병목 없음 → 손해 소멸.","fig11b_sides_fixed.svg")

# fig13: makespan vs N (수정후, fig9 스타일) — ALL-GPU vs SCHEDULED
def fig_ms_fixed():
    allgpu=[576.43,747.72,964.40,1112.99,1391.61]; sched=[425.65,747.38,935.31,1027.27,1189.77]
    W,H=720,430; b=[text(20,28,"[Fig 13] (수정 후) N별 makespan — ALL-GPU vs SCHEDULED",15,weight="bold")]
    x0=70;y0=55;pw=600;ph=290;mx=1500
    def sx(n): return x0+pw*(n-40)/(130-40)
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,1501,300):
        b.append(line(x0,sy(gv),x0+pw,sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv}",10,"end",fill="#999"))
    for n in N: b.append(line(sx(n),y0,sx(n),y0+ph,GRID)); b.append(text(sx(n),y0+ph+18,f"{n}",11,"middle"))
    b.append(text(x0+pw/2,y0+ph+38,"task 수 N",12,"middle"))
    def poly(vals,col):
        pts=" ".join(f"{sx(N[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(N)))
        s=f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>'
        for i in range(len(N)): s+=circ(sx(N[i]),sy(vals[i]),4,col)
        return s
    # 절감 영역(두 선 사이) 음영
    pts=[f"{sx(N[i]):.1f},{sy(allgpu[i]):.1f}" for i in range(len(N))]+[f"{sx(N[i]):.1f},{sy(sched[i]):.1f}" for i in range(len(N)-1,-1,-1)]
    b.append(f'<polygon points="{" ".join(pts)}" fill="{GREEN}" opacity="0.12"/>')
    b.append(poly(allgpu,GPU_COL)); b.append(poly(sched,GREEN))
    for i in range(len(N)):
        sv=allgpu[i]/sched[i]
        b.append(text(sx(N[i]),sy(sched[i])+16,f"{sv:.2f}x",10,"middle",weight="bold",fill=GREEN))
    b.append(rect(x0+pw-150,y0+6,14,10,GPU_COL)); b.append(text(x0+pw-132,y0+15,"ALL-GPU",11))
    b.append(rect(x0+pw-150,y0+24,14,10,GREEN)); b.append(text(x0+pw-132,y0+33,"SCHEDULED(수정후)",11))
    b.append(text(x0,H-14,"SCHEDULED(초록)가 ALL-GPU(빨강) 아래로 유지 → 초록 음영=절감. N=120서 격차 최대(1.17x).",11,fill=GREEN))
    save("fig13_ms_fixed.svg",svg(W,H,"".join(b)))
fig_ms_fixed()

# ---- 모든 기존 SVG에 XML 선언 추가(없으면) ----
for p in glob.glob(os.path.join(FIG,"*.svg")):
    d=open(p,encoding="utf-8").read()
    if not d.lstrip().startswith("<?xml"):
        open(p,"w",encoding="utf-8").write(XMLDECL+d); print("xml-decl +",os.path.basename(p))
print("DONE")
