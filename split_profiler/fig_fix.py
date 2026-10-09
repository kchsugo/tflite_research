# -*- coding: utf-8 -*-
# 보수적 offload 수정 전/후 비교 + 수정후 GPU측/CPU측 → docs/figures/*.svg
import os,sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
TXT="#222"; GRID="#ddd"; GPU_COL="#e15759"; CPU_COL="#4e79a7"; GREEN="#59a14f"; GREY="#999"; BLACK="#000"
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def circ(x,y,r,fill): return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'
def rect(x,y,w,h,fill,op=1): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(0,h):.1f}" fill="{fill}" opacity="{op}"/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

N=[50,70,90,100,120]
before=[1.072,1.021,1.000,0.927,0.858]   # 수정 전(과잉 offload)
after =[1.354,1.000,1.031,1.083,1.170]   # 수정 후(CPU측이 GPU측 넘으면 취소)

# ---- Fig 12: 수정 전/후 speedup ----
def fig_ba():
    W,H=760,410; b=[text(20,28,"[Fig 12] 보수적 offload 수정 전/후 speedup (vs ALL-GPU)",15,weight="bold")]
    x0=70;y0=55;pw=620;ph=280;mxv=1.4;mnv=0.8
    def sx(n): return x0+pw*(n-40)/(130-40)
    def sy(v): return y0+ph*(mxv-v)/(mxv-mnv)
    b.append(rect(x0,y0,pw,sy(1.0)-y0,GREEN,0.06)); b.append(rect(x0,sy(1.0),pw,y0+ph-sy(1.0),GPU_COL,0.06))
    for gv in [0.8,0.9,1.0,1.1,1.2,1.3,1.4]:
        b.append(line(x0,sy(gv),x0+pw,sy(gv),BLACK if gv==1.0 else GRID,1.5 if gv==1.0 else 1,"4" if gv==1.0 else ""))
        b.append(text(x0-8,sy(gv)+4,f"{gv:.1f}",10,"end",fill="#999"))
    for n in N: b.append(text(sx(n),y0+ph+18,f"{n}",11,"middle"))
    b.append(text(x0+pw/2,y0+ph+38,"task 수 N",12,"middle"))
    def poly(vals,col,dash=""):
        pts=" ".join(f"{sx(N[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(N)))
        s=f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"{" stroke-dasharray="+chr(34)+"6"+chr(34) if dash else ""}/>'
        for i in range(len(N)): s+=circ(sx(N[i]),sy(vals[i]),4,col)
        return s
    b.append(poly(before,GREY,"d")); b.append(poly(after,GREEN))
    for i in range(len(N)): b.append(text(sx(N[i]),sy(after[i])-9,f"{after[i]:.2f}",10,"middle",weight="bold",fill=GREEN))
    b.append(line(x0+pw-170,y0+6,x0+pw-150,y0+6,GREY,2.5,"6")); b.append(text(x0+pw-146,y0+10,"수정 전(과잉 offload)",11))
    b.append(line(x0+pw-170,y0+24,x0+pw-150,y0+24,GREEN,2.5)); b.append(text(x0+pw-146,y0+28,"수정 후(CPU>GPU면 취소)",11))
    b.append(text(x0,H-12,"수정 후: 전부 1.0x 이상(손해 소멸), N 커질수록 이득↑ (N=120: 0.86x→1.17x).",11,fill=GREEN))
    save("fig12_fix_before_after.svg",svg(W,H,"".join(b)))

# ---- Fig 11 갱신: 수정 후 GPU측/CPU측 (CPU측이 항상 GPU측 아래) ----
def fig_sides_new():
    gpu_side=[425.55,747.34,935.27,1027.24,1189.73]
    cpu_side=[263.90,423.21,784.98,824.73,800.71]
    all_gpu =[576.43,747.72,964.40,1112.99,1391.61]
    W,H=780,440; b=[text(20,28,"[Fig 11] (수정 후) N별 GPU측 vs CPU측. CPU측이 항상 GPU측 아래 → 손해 없음",15,weight="bold")]
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
    b.append(rect(x0,H-46,14,12,GPU_COL)); b.append(text(x0+20,H-36,"GPU측(=makespan)",11))
    b.append(rect(x0+180,H-46,14,12,CPU_COL)); b.append(text(x0+200,H-36,"CPU측(항상 아래)",11))
    b.append(line(x0+360,H-40,x0+380,H-40,BLACK,2)); b.append(text(x0+386,H-36,"ALL-GPU",11))
    b.append(text(x0,H-14,"CPU측(파랑)이 GPU측(빨강)보다 항상 낮음 → CPU가 병목 안 됨. GPU측<ALL-GPU선 → 이득.",11,fill=GREEN))
    save("fig11_sides.svg",svg(W,H,"".join(b)))

fig_ba(); fig_sides_new(); print("DONE")
