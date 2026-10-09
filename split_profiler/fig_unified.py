# -*- coding: utf-8 -*-
# 통합 플래너 결과 시각화 → docs/figures/figU*.svg  (무의존성 순수 SVG)
import os, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
GPU="#2E7D32"; CPU="#2E5FA0"; SPL="#7B4EA8"; REF="#9AA0A6"; HARM="#C00000"; TXT="#222"; GRID="#ddd"
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,st=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{st}" stroke-width="{w}"{d}/>'
def rect(x,y,w,h,fill,op=1,stroke=None):
    s=f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" fill="{fill}" opacity="{op}"{s}/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

# name, 구성, gpu, cpu, split, allgpu, unified, speedup, gpu측, cpu측
W=[
 ("N=50 #1","mob4 lan7 bise16 yolo10 wide13",41,9,0,710.6,679.2,1.05,679.2,296.1),
 ("N=50 #2","mob12 lan10 bise11 yolo4 wide13",34,16,0,432.1,420.2,1.03,420.2,159.4),
 ("N=50 #3","mob7 lan9 bise12 yolo8 wide14",38,12,0,599.3,536.7,1.12,536.6,358.0),
 ("N=100 #1","mob12 lan20 bise25 yolo19 wide24",78,22,0,1145.7,1149.4,1.00,1149.3,629.5),
 ("N=100 #2","mob27 lan17 bise27 yolo11 wide18",70,30,0,1210.9,1080.8,1.12,1080.8,804.6),
]
LAB=[w[0] for w in W]

# ---- figU1: speedup ----
def figU1():
    Wd,H=760,380; b=[text(20,28,"[U1] 워크로드별 speedup (vs ALL-GPU) — 전부 ≥ 1.0x (손해 없음)",15,weight="bold")]
    x0=70;y0=70;ph=230;n=len(W);slot=(Wd-x0-30)/n;bw=slot*0.45
    mx=1.25
    def sy(v): return y0+ph*(1-(v-0.9)/(mx-0.9))
    for g in [0.9,1.0,1.1,1.2]:
        b.append(line(x0,sy(g),Wd-30,sy(g),GRID if g!=1.0 else HARM,1 if g!=1.0 else 1.5,"" if g!=1.0 else "4"))
        b.append(text(x0-6,sy(g)+4,f"{g:.1f}",10,"end",fill="#999"))
    b.append(text(Wd-32,sy(1.0)-4,"1.0 (ALL-GPU)",10,"end",fill=HARM))
    for i,w in enumerate(W):
        cx=x0+slot*i+slot/2; sp=w[7]
        b.append(rect(cx-bw/2,sy(sp),bw,y0+ph-sy(sp),GPU if sp>=1.0 else HARM))
        b.append(text(cx,sy(sp)-6,f"{sp:.2f}x",12,"middle",weight="bold",fill=GPU))
        b.append(text(cx,y0+ph+18,LAB[i],11,"middle"))
    save("figU1_speedup.svg",svg(Wd,H,"".join(b)))

# ---- figU2: 배정 (GPU/CPU/SPLIT) ----
def figU2():
    Wd,H=760,400; b=[text(20,28,"[U2] 모드별 배정 — 통짜-GPU / 통짜-CPU / 분할(SPLIT)",15,weight="bold")]
    b.append(text(20,47,"SPLIT=0 → fp32에선 전부 '통짜'로 배정(분할 안 됨). CPU는 필요한 만큼만.",11,fill="#555"))
    x0=70;y0=75;ph=225;n=len(W);slot=(Wd-x0-30)/n;bw=slot*0.5
    mx=max(w[2]+w[3]+w[4] for w in W)*1.1
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,20):
        b.append(line(x0,sy(gv),Wd-30,sy(gv),GRID));b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"task 수",10,fill="#999"))
    for i,w in enumerate(W):
        cx=x0+slot*i+slot/2; g,c,s=w[2],w[3],w[4]; base=y0+ph
        hg=ph*g/mx; b.append(rect(cx-bw/2,base-hg,bw,hg,GPU)); b.append(text(cx,base-hg/2+4,str(g),11,"middle",fill="white",weight="bold"))
        hc=ph*c/mx; b.append(rect(cx-bw/2,base-hg-hc,bw,hc,CPU)); b.append(text(cx,base-hg-hc/2+4,str(c),11,"middle",fill="white",weight="bold"))
        if s>0: hs=ph*s/mx; b.append(rect(cx-bw/2,base-hg-hc-hs,bw,hs,SPL))
        b.append(text(cx,base+18,LAB[i],11,"middle"))
    lx=x0;ly=H-18
    for dx,col,t in [(0,GPU,"통짜-GPU"),(110,CPU,"통짜-CPU"),(220,SPL,"분할(=0)")]:
        b.append(rect(lx+dx,ly-10,12,12,col));b.append(text(lx+dx+16,ly,t,11))
    save("figU2_assign.svg",svg(Wd,H,"".join(b)))

# ---- figU3: makespan (ALL-GPU vs UNIFIED) ----
def figU3():
    Wd,H=760,400; b=[text(20,28,"[U3] makespan: ALL-GPU vs 통합 플래너 (ms, 낮을수록 빠름)",15,weight="bold")]
    x0=70;y0=70;ph=250;n=len(W);slot=(Wd-x0-30)/n;bw=slot*0.28
    mx=max(w[5] for w in W)*1.12
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,200):
        b.append(line(x0,sy(gv),Wd-30,sy(gv),GRID));b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"ms",10,fill="#999"))
    for i,w in enumerate(W):
        cx=x0+slot*i+slot/2; ag,un=w[5],w[6]
        b.append(rect(cx-bw-2,sy(ag),bw,y0+ph-sy(ag),REF)); b.append(text(cx-bw/2-2,sy(ag)-4,f"{ag:.0f}",9,"middle",fill="#777"))
        b.append(rect(cx+2,sy(un),bw,y0+ph-sy(un),GPU)); b.append(text(cx+bw/2+2,sy(un)-4,f"{un:.0f}",9,"middle",fill=GPU))
        saved=(1-un/ag)*100
        b.append(text(cx,sy(min(ag,un))-18,f"-{saved:.0f}%",11,"middle",weight="bold",fill=HARM if saved>1 else "#999"))
        b.append(text(cx,y0+ph+18,LAB[i],11,"middle"))
    lx=x0;ly=H-14
    b.append(rect(lx,ly-10,12,12,REF));b.append(text(lx+16,ly,"ALL-GPU",11))
    b.append(rect(lx+120,ly-10,12,12,GPU));b.append(text(lx+136,ly,"통합 플래너",11))
    save("figU3_makespan.svg",svg(Wd,H,"".join(b)))

# ---- figU4: GPU측 vs CPU측 (사용률 균형) ----
def figU4():
    Wd,H=760,400; b=[text(20,28,"[U4] 자원 사용률 — GPU측 vs CPU측 완료시각 (균형 지표)",15,weight="bold")]
    b.append(text(20,47,"두 막대가 비슷할수록 CPU/GPU 균등 활용. CPU측이 GPU측을 넘지 않아 손해 없음.",11,fill="#555"))
    x0=70;y0=75;ph=225;n=len(W);slot=(Wd-x0-30)/n;bw=slot*0.28
    mx=max(w[8] for w in W)*1.12
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,200):
        b.append(line(x0,sy(gv),Wd-30,sy(gv),GRID));b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"ms",10,fill="#999"))
    for i,w in enumerate(W):
        cx=x0+slot*i+slot/2; gs,cs=w[8],w[9]
        b.append(rect(cx-bw-2,sy(gs),bw,y0+ph-sy(gs),GPU)); b.append(text(cx-bw/2-2,sy(gs)-4,f"{gs:.0f}",9,"middle",fill=GPU))
        b.append(rect(cx+2,sy(cs),bw,y0+ph-sy(cs),CPU)); b.append(text(cx+bw/2+2,sy(cs)-4,f"{cs:.0f}",9,"middle",fill=CPU))
        bal=cs/gs*100
        b.append(text(cx,y0+ph+18,f"{LAB[i]}",11,"middle"))
        b.append(text(cx,y0+ph+32,f"균형{bal:.0f}%",9,"middle",fill="#777"))
    lx=x0;ly=H-14
    b.append(rect(lx,ly-10,12,12,GPU));b.append(text(lx+16,ly,"GPU측",11))
    b.append(rect(lx+90,ly-10,12,12,CPU));b.append(text(lx+106,ly,"CPU측",11))
    save("figU4_balance.svg",svg(Wd,H,"".join(b)))

# ---- figU5: 스케줄링 도식 (대표: N=50 #3) ----
def figU5():
    w=W[2]  # N=50 #3: gpu38 cpu12, gpu측536.6 cpu측358
    Wd,H=820,420; b=[text(20,28,"[U5] 스케줄링 구조 (예: N=50 #3, 38 GPU / 12 CPU, 분할 0)",15,weight="bold")]
    b.append(text(20,47,"GPU=메인 1레인 직렬,  CPU=P-1(=5)코어 병렬(1코어는 GPU 구동 예약).  각 task=통짜(안 쪼갬).",11,fill="#555"))
    x0=110; xw=650; t_max=w[8]  # gpu측
    def sx(t): return x0+xw*t/t_max
    # GPU 레인
    y=80; b.append(text(20,y+14,"GPU",12,weight="bold",fill=GPU)); b.append(text(20,y+30,"(직렬)",9,fill="#777"))
    b.append(rect(x0,y,xw,26,GPU,0.85)); b.append(text(x0+xw/2,y+17,"통짜-GPU task 38개 (직렬 실행)",11,"middle",fill="white",weight="bold"))
    b.append(text(x0+xw+6,y+17,f"{w[8]:.0f}ms",10,fill=GPU))
    # CPU 코어들
    cores=5; per=w[3]/cores
    for cidx in range(cores):
        yy=130+cidx*40; ntasks=[3,3,2,2,2][cidx]
        b.append(text(20,yy+16,f"CPU{cidx+1}",11,fill=CPU))
        frac=w[9]/t_max  # cpu측 길이 비율
        b.append(rect(x0,yy,xw*frac,26,CPU,0.8))
        b.append(text(x0+xw*frac/2,yy+17,f"통짜-CPU {ntasks}개",10,"middle",fill="white"))
    b.append(text(x0+xw*(w[9]/t_max)+6,130+16,f"{w[9]:.0f}ms",10,fill=CPU))
    # makespan 선
    b.append(line(sx(t_max),70,sx(t_max),340,HARM,1.5,"4"))
    b.append(text(sx(t_max)+4,335,f"makespan={w[6]:.0f}ms = max(GPU측,CPU측)",10,fill=HARM))
    # 설명
    b.append(text(20,375,"● 분할(SPLIT) 없음 — 모든 task는 통짜로 GPU 또는 CPU에 통째 배정",11,fill=SPL))
    b.append(text(20,393,"● CPU측(358) < GPU측(537) → CPU가 병목 안 됨 → GPU 병목을 덜어 1.12x",11,fill="#555"))
    save("figU5_schedule.svg",svg(Wd,H,"".join(b)))

if __name__=="__main__":
    figU1(); figU2(); figU3(); figU4(); figU5()
    print("done")
