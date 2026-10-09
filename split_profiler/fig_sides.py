# -*- coding: utf-8 -*-
# N별 GPU측 vs CPU측 완료시간 → docs/figures/fig11_sides.svg
import os,sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
TXT="#222"; GRID="#ddd"; GPU_COL="#e15759"; CPU_COL="#4e79a7"; BLACK="#000"
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def rect(x,y,w,h,fill): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(0,h):.1f}" fill="{fill}"/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'

N=[50,70,90,100,120]
gpu_side=[480.55,678.24,883.90,1046.81,1211.32]
cpu_side=[446.59,804.97,1095.77,1103.87,1476.14]
all_gpu =[614.00,857.87,1019.47,1219.88,1356.11]

W,H=780,440; b=[text(20,28,"[Fig 11] N별 GPU측 vs CPU측 완료시간 (통짜 SCHEDULED). 검은선=ALL-GPU 기준",15,weight="bold")]
x0=70;y0=55;pw=640;ph=300;mx=1600;grp=pw/len(N);bw=42
def sy(v): return y0+ph*(1-v/mx)
for gv in range(0,1601,400):
    b.append(line(x0,sy(gv),x0+pw,sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv}",10,"end",fill="#999"))
for i,n in enumerate(N):
    cx=x0+grp*i+grp/2
    # GPU측 / CPU측 막대
    b.append(rect(cx-bw-3,sy(gpu_side[i]),bw,y0+ph-sy(gpu_side[i]),GPU_COL))
    b.append(rect(cx+3,sy(cpu_side[i]),bw,y0+ph-sy(cpu_side[i]),CPU_COL))
    b.append(text(cx-bw/2-3,sy(gpu_side[i])-4,f"{gpu_side[i]:.0f}",9,"middle",fill=GPU_COL))
    b.append(text(cx+bw/2+3,sy(cpu_side[i])-4,f"{cpu_side[i]:.0f}",9,"middle",fill=CPU_COL))
    # ALL-GPU 기준선 (그 그룹 위 짧은 검은 선)
    b.append(line(cx-bw-6,sy(all_gpu[i]),cx+bw+6,sy(all_gpu[i]),BLACK,2))
    b.append(text(cx,y0+ph+18,f"N={n}",11,"middle"))
    # 병목 표시
    bott = "CPU측 병목" if cpu_side[i]>gpu_side[i] else "GPU측 병목"
    col = CPU_COL if cpu_side[i]>gpu_side[i] else GPU_COL
    b.append(text(cx,y0+ph+34,bott,9,"middle",fill=col))
# 범례
b.append(rect(x0,H-46,14,12,GPU_COL)); b.append(text(x0+20,H-36,"GPU측 완료시간",11))
b.append(rect(x0+150,H-46,14,12,CPU_COL)); b.append(text(x0+170,H-36,"CPU측 완료시간",11))
b.append(line(x0+300,H-40,x0+320,H-40,BLACK,2)); b.append(text(x0+326,H-36,"ALL-GPU (기준)",11))
b.append(text(x0,H-14,"GPU측은 offload할수록 줄어듦(경합X). N↑에서 CPU측이 ALL-GPU선을 넘어 병목→손해.",11,fill="#a00"))
open(os.path.join(FIG,"fig11_sides.svg"),"w",encoding="utf-8").write(svg(W,H,"".join(b)))
print("wrote fig11_sides.svg")
