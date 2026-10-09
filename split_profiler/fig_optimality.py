# -*- coding: utf-8 -*-
# ============================================================================
#  최적성(optimality) 그림 세트 → docs/figures/figA..D_*.svg
# ----------------------------------------------------------------------------
#  목적: "ALL-GPU 대비"(환경 종속)가 아니라 "주어진 워크로드에서 이론적 하한(LB)에
#        얼마나 가깝게 분할·배치·정렬했는가"를 보인다.
#
#  LB(W) = max( CPUside(W), GPUside(W) )      # 기계 1개라 그 방향 총작업량이 바닥선
#  CPUside = Σ cpu_stage_i,  GPUside = Σ gpu_stage_i   (온라인이 고른 방향의 stage)
#
#  데이터:
#   - stage/방향/LB : split_profile.csv (오프라인 측정, 계산으로 확정)
#   - makespan_mine : jackson_online 실측(ms). MEAS[...] 에 반복 측정값 그대로.
#                     (build+warmup 제외한 순수 Invoke 구간; LB는 전송/경합/warmup 제외)
#  환경: RTX 3060 Laptop(6GB,OpenCL) + i5-11400H(6C) / TFLite fp32 / threads=4
# ============================================================================
import os, sys, csv
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
CSV=os.path.join(HERE,"split_profile.csv")

# ---- 팔레트 (스펙 지정) ----
TXT="#222"; GRID="#ddd"
CPU_COL="#2E5FA0"   # CPU (파랑)
GPU_COL="#2E7D32"   # GPU (초록)
OPT="#7B4EA8"       # 강조/내 스케줄 (보라)
HARM="#C00000"      # 경계/손해 (빨강)
REF="#9AA0A6"       # 하한/기준선 (회색)

def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def rect(x,y,w,h,fill,op=1,stroke=None):
    s=f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" fill="{fill}" opacity="{op}"{s}/>'
def svg(w,h,b): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s); print("wrote",n)

# ---- (1) split_profile.csv → 모델별 방향별 stage ----
P={}   # name -> {'CPU-first':(k,cpu,gpu), 'GPU-first':..., 'ALL-GPU':gwhole}
with open(CSV,newline='') as f:
    for r in csv.DictReader(f):
        nm=r['model']; d=r['direction']
        cpu=float(r['cpu_stage_ms']); gpu=float(r['gpu_stage_ms'])
        P.setdefault(nm,{})[d]=(int(r['k']),cpu,gpu)

# 온라인 방향 선택 규칙: max(cpu,gpu) 작은 쪽
def chosen(nm):
    _,cf_c,cf_g=P[nm]['CPU-first']; _,gf_c,gf_g=P[nm]['GPU-first']
    if max(gf_c,gf_g)<max(cf_c,cf_g): return ('GPU-first',gf_c,gf_g)
    return ('CPU-first',cf_c,cf_g)

MOB="mobilenet_v2_full.tflite"; LAN="lanenet_fp32.tflite"; BISE="BiseNet.tflite"
YOLO="yolov4-tiny-416_ieie (1).tflite"; WIDE="WideResNet50_float.tflite"

# ---- (2) 워크로드 정의 (fig4/fig6과 동일 이름·순서) ----
#   각 원소 (모델, 개수). 라벨 2줄.
WL=[
 ("mob2+yolo2\n(4, mixed)",      [(MOB,2),(YOLO,2)]),
 ("each×2\n(10, mixed)",         [(MOB,2),(LAN,2),(BISE,2),(YOLO,2),(WIDE,2)]),
 ("each×3\n(15, mixed)",         [(MOB,3),(LAN,3),(BISE,3),(YOLO,3),(WIDE,3)]),
 ("mob×6\n(6, same-dir)",        [(MOB,6)]),
 ("bise×4\n(4, same-dir)",       [(BISE,4)]),
]

# ---- (3) 실측 makespan (jackson_online, ms). 반복 측정값 그대로(평균/min/max 계산) ----
#   측정: 2026-09  (build+warmup 제외 구간). all_gpu = ALL-GPU baseline 실측.
MEAS={
 "mob2+yolo2\n(4, mixed)": dict(mine=[12.08,9.47,9.86,10.80], allgpu=[10.61,10.80,10.43,10.37]),
 "each×2\n(10, mixed)":    dict(mine=[83.49,77.25],           allgpu=[79.39,78.89]),
 "each×3\n(15, mixed)":    dict(mine=[120.51,118.56],         allgpu=[115.37,108.88]),
 "mob×6\n(6, same-dir)":   dict(mine=[8.66,8.62,8.77,9.04,8.85,9.07,8.62], allgpu=[9.00,9.14,8.93,9.88,9.29,8.91,8.50]),
 "bise×4\n(4, same-dir)":  dict(mine=[90.97,86.67,94.64,98.58,85.15,84.93,85.78], allgpu=[84.67,79.74,83.16,83.21,79.21,79.08,78.84]),
}
def avg(v): return sum(v)/len(v)

# ---- (4) 워크로드별 지표 계산 ----
rows=[]
for lab,comp in WL:
    cpus=gpus=0.0
    for nm,cnt in comp:
        _,c,g=chosen(nm); cpus+=cnt*c; gpus+=cnt*g
    LB=max(cpus,gpus)
    m=MEAS[lab]; mine=avg(m['mine']); ag=avg(m['allgpu'])
    imb=abs(cpus-gpus)/max(cpus,gpus)*100
    gap=(mine/LB-1)*100
    rows.append(dict(lab=lab,cpu=cpus,gpu=gpus,LB=LB,mine=mine,allgpu=ag,
                     mine_lo=min(m['mine']),mine_hi=max(m['mine']),imb=imb,gap=gap))

# ============================================================================
#  Fig A — Optimality Gap : makespan_mine vs 이론 하한 LB
# ============================================================================
def figA():
    W,H=820,430; b=[text(20,28,"[Fig A] Optimality Gap — 실측 makespan vs 이론 하한 LB=max(ΣCPU,ΣGPU)",15,weight="bold")]
    b.append(text(20,48,"회색=이론 하한(LB, 전송·경합·warmup 제외)  보라=Jackson 실측(오차막대=측정범위).  gap%=(실측/LB−1)",11,fill="#555"))
    x0=70; y0=80; ph=250; n=len(rows); slot=(W-x0-30)/n; bw=slot*0.30
    mx=max(r['mine'] for r in rows)*1.15
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,20):
        b.append(line(x0,sy(gv),W-30,sy(gv),GRID)); b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"ms",10,fill="#999"))
    for i,r in enumerate(rows):
        cx=x0+slot*i+slot/2
        # LB 막대(회색) + mine 막대(보라)
        b.append(rect(cx-bw-2,sy(r['LB']),bw,y0+ph-sy(r['LB']),REF))
        b.append(rect(cx+2,sy(r['mine']),bw,y0+ph-sy(r['mine']),OPT))
        # 오차막대(측정 범위)
        ex=cx+2+bw/2
        b.append(line(ex,sy(r['mine_lo']),ex,sy(r['mine_hi']),"#333",1.5))
        b.append(line(ex-4,sy(r['mine_lo']),ex+4,sy(r['mine_lo']),"#333",1.5))
        b.append(line(ex-4,sy(r['mine_hi']),ex+4,sy(r['mine_hi']),"#333",1.5))
        # gap 라벨
        col=HARM if r['gap']>7 else OPT
        b.append(text(cx,sy(max(r['mine'],r['LB']))-8,f"{r['gap']:+.0f}%",12,"middle",weight="bold",fill=col))
        b.append(text(cx,y0+ph+16,f"LB {r['LB']:.0f}",10,"middle",fill="#777"))
        for j,ln in enumerate(r['lab'].split("\n")):
            b.append(text(cx,y0+ph+30+j*13,ln,10,"middle",fill="#444"))
    # 범례
    lx=x0; ly=H-22
    b.append(rect(lx,ly-10,12,12,REF)); b.append(text(lx+16,ly,"이론 하한 LB",11))
    b.append(rect(lx+140,ly-10,12,12,OPT)); b.append(text(lx+156,ly,"Jackson 실측 makespan",11))
    b.append(text(lx+360,ly,"gap 작을수록 최적에 가까움 (±5~7%는 노이즈)",11,fill=HARM))
    save("figA_optimality_gap.svg", svg(W,H,"".join(b)))

# ============================================================================
#  Fig B — 두 레인 균형 : CPUside vs GPUside (배치가 얼마나 균형인가)
# ============================================================================
def figB():
    W,H=820,430; b=[text(20,28,"[Fig B] 두 레인 균형 — CPUside(ΣCPU-part) vs GPUside(ΣGPU-part)",15,weight="bold")]
    b.append(text(20,48,"막대 길이=각 기계 총작업량(ms). makespan 하한=긴 쪽. 불균형%가 클수록 겹칠 여지 없음→GPU-bound.",11,fill="#555"))
    x0=70; y0=80; ph=250; n=len(rows); slot=(W-x0-30)/n; bw=slot*0.30
    mx=max(r['gpu'] for r in rows)*1.15
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,20):
        b.append(line(x0,sy(gv),W-30,sy(gv),GRID)); b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"ms",10,fill="#999"))
    for i,r in enumerate(rows):
        cx=x0+slot*i+slot/2
        b.append(rect(cx-bw-2,sy(r['cpu']),bw,y0+ph-sy(r['cpu']),CPU_COL))
        b.append(rect(cx+2,sy(r['gpu']),bw,y0+ph-sy(r['gpu']),GPU_COL))
        b.append(text(cx-bw/2-2,sy(r['cpu'])-4,f"{r['cpu']:.1f}",9,"middle",fill=CPU_COL))
        b.append(text(cx+bw/2+2,sy(r['gpu'])-4,f"{r['gpu']:.1f}",9,"middle",fill=GPU_COL))
        b.append(text(cx,sy(r['gpu'])-20,f"불균형 {r['imb']:.0f}%",11,"middle",weight="bold",fill=HARM))
        for j,ln in enumerate(r['lab'].split("\n")):
            b.append(text(cx,y0+ph+16+j*13,ln,10,"middle",fill="#444"))
    lx=x0; ly=H-22
    b.append(rect(lx,ly-10,12,12,CPU_COL)); b.append(text(lx+16,ly,"CPUside",11))
    b.append(rect(lx+90,ly-10,12,12,GPU_COL)); b.append(text(lx+106,ly,"GPUside",11))
    b.append(text(lx+200,ly,"CPU가 4~100배 느려 균형 cut이 GPU로 치우침 → 불균형 극심 → 겹침 이득 거의 없음",11,fill="#555"))
    save("figB_lane_balance.svg", svg(W,H,"".join(b)))

# ============================================================================
#  Fig C — LB / Jackson실측 / ALL-GPU  (내 스케줄이 하한에 얼마나 가깝고 naive를 이기나)
# ============================================================================
def figC():
    W,H=820,430; b=[text(20,28,"[Fig C] 이론하한 vs Jackson실측 vs ALL-GPU(naive)",15,weight="bold")]
    b.append(text(20,48,"회색=LB(하한)  보라=Jackson 실측  빨강=ALL-GPU(통짜 GPU 직렬). 셋이 붙을수록 '이 워크로드는 순서 무관·GPU-bound'.",11,fill="#555"))
    x0=70; y0=80; ph=250; n=len(rows); slot=(W-x0-30)/n; bw=slot*0.22
    mx=max(max(r['mine'],r['allgpu']) for r in rows)*1.15
    def sy(v): return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,20):
        b.append(line(x0,sy(gv),W-30,sy(gv),GRID)); b.append(text(x0-6,sy(gv)+4,str(gv),10,"end",fill="#999"))
    b.append(text(20,y0-4,"ms",10,fill="#999"))
    for i,r in enumerate(rows):
        cx=x0+slot*i+slot/2
        for k,(v,c) in enumerate([(r['LB'],REF),(r['mine'],OPT),(r['allgpu'],HARM)]):
            bx=cx+(k-1)*(bw+3)-bw/2
            b.append(rect(bx,sy(v),bw,y0+ph-sy(v),c))
            b.append(text(bx+bw/2,sy(v)-3,f"{v:.0f}",8,"middle",fill=c))
        for j,ln in enumerate(r['lab'].split("\n")):
            b.append(text(cx,y0+ph+16+j*13,ln,10,"middle",fill="#444"))
    lx=x0; ly=H-22
    for dx,c,t in [(0,REF,"LB"),(90,OPT,"Jackson"),(220,HARM,"ALL-GPU")]:
        b.append(rect(lx+dx,ly-10,12,12,c)); b.append(text(lx+dx+16,ly,t,11))
    b.append(text(lx+340,ly,"random 순서 baseline은 미측정(추가 실행 필요) — 여기선 naive=ALL-GPU로 대체",11,fill="#555"))
    save("figC_lb_mine_allgpu.svg", svg(W,H,"".join(b)))

# ============================================================================
#  Fig D — 분할 품질 : 각 모델 1컷의 stage 균형 (선택된 방향)
# ============================================================================
def figD():
    models=[MOB,LAN,YOLO,WIDE,BISE]
    short={MOB:"mobilenet",LAN:"lanenet",YOLO:"yolov4-tiny",WIDE:"WideResNet50",BISE:"BiseNet"}
    W,H=820,60+len(models)*54; b=[text(20,28,"[Fig D] 분할 품질 — 각 모델 1컷의 두 stage 균형(선택된 방향)",15,weight="bold")]
    b.append(text(20,48,"파랑=CPU-stage, 초록=GPU-stage(ms). stage 불균형%가 작을수록 그 task 내부 cut이 최적.",11,fill="#555"))
    x0=170; xw=470; y=68
    mx=max(max(chosen(m)[1],chosen(m)[2]) for m in models) or 1
    def sx(v): return xw*(v/mx)
    for m in models:
        d,c,g=chosen(m); imb=abs(c-g)/max(c,g,1e-9)*100
        b.append(text(20,y+16,short[m],12,weight="bold")); b.append(text(20,y+30,d,10,fill="#777"))
        b.append(rect(x0,y,sx(c),13,CPU_COL)); b.append(text(x0+sx(c)+4,y+11,f"{c:.2f}",9,fill=CPU_COL))
        b.append(rect(x0,y+15,sx(g),13,GPU_COL)); b.append(text(x0+sx(g)+4,y+26,f"{g:.2f}",9,fill=GPU_COL))
        col=HARM if imb>50 else OPT
        b.append(text(x0+xw+70,y+18,f"불균형 {imb:.0f}%",11,"start",weight="bold",fill=col))
        y+=54
    save("figD_split_quality.svg", svg(W,H,"".join(b)))

if __name__=="__main__":
    figA(); figB(); figC(); figD()
    print("\n== 워크로드별 요약 ==")
    print(f"{'workload':22s} {'CPUside':>8s} {'GPUside':>8s} {'LB':>7s} {'mine':>7s} {'gap%':>6s} {'imb%':>6s}")
    for r in rows:
        print(f"{r['lab'].replace(chr(10),' '):22s} {r['cpu']:8.2f} {r['gpu']:8.2f} {r['LB']:7.2f} {r['mine']:7.2f} {r['gap']:+6.1f} {r['imb']:6.1f}")
