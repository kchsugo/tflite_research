# -*- coding: utf-8 -*-
# 의존성 없이 순수 Python으로 SVG 그래프 생성 → docs/figures/*.svg
import csv, os, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG  = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
CSV  = os.path.join(HERE, "split_profile.csv")

CPU_COL="#4e79a7"; GPU_COL="#e15759"; GRID="#dddddd"; TXT="#222222"
MCOL={"mob":"#59a14f","lan":"#4e79a7","bise":"#e15759","yolo":"#f28e2b","wide":"#b07aa1"}
ALIAS={"mob":"mobilenet_v2_full.tflite","lan":"lanenet_fp32.tflite","bise":"BiseNet.tflite",
       "yolo":"yolov4-tiny-416_ieie (1).tflite","wide":"WideResNet50_float.tflite"}
SHORT={v:k for k,v in ALIAS.items()}

def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def rect(x,y,w,h,fill,stroke="none",rx=0):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" rx="{rx}"/>'
def text(x,y,s,size=13,anchor="start",fill=TXT,weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI, Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>'
def line(x1,y1,x2,y2,stroke=GRID,w=1,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'
def svg(w,h,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{body}</svg>'
def save(name,s):
    with open(os.path.join(FIG,name),"w",encoding="utf-8") as f: f.write(s)
    print("wrote", name)

# ---- 데이터 로드 ----
P={}
for r in csv.DictReader(open(CSV,newline='')):
    nm=r['model']; d=r['direction']; P.setdefault(nm,{})
    row={'k':int(r['k']),'cpu':float(r['cpu_stage_ms']),'gpu':float(r['gpu_stage_ms']),
         'cn':int(r['cpu_nodes']),'gn':int(r['gpu_nodes'])}
    P[nm]={**P[nm], {'CPU-first':'cf','GPU-first':'gf','ALL-GPU':'ag','ALL-CPU':'ac'}.get(d,d):row}
models=[m for m in ["mobilenet_v2_full.tflite","lanenet_fp32.tflite","BiseNet.tflite",
        "yolov4-tiny-416_ieie (1).tflite","WideResNet50_float.tflite"] if m in P]

# =========================================================================
# Fig 1: 모델별 절단 지점 (CPU-first / GPU-first)
# =========================================================================
def fig_cuts():
    W,H=820,60+len(models)*70; b=[]
    b.append(text(20,30,"[Fig 1] 모델별 절단 지점 (파랑=CPU 노드, 빨강=GPU 노드, | =cut)",16,weight="bold"))
    y=55; barx=150; barw=560
    for m in models:
        s=SHORT[m]
        for d,lab in (('cf','CPU-first'),('gf','GPU-first')):
            r=P[m][d]; tot=r['cn']+r['gn']
            if tot==0:
                y+=30; continue
            b.append(text(20,y+15,f"{s}",14,weight="bold"))
            b.append(text(70,y+15,f"{lab} k={r['k']}",12,fill="#666"))
            if d=='cf':  # 앞 CPU, 뒤 GPU
                cw=barw*r['cn']/tot
                b.append(rect(barx,y,cw,20,CPU_COL))
                b.append(rect(barx+cw,y,barw-cw,20,GPU_COL))
                b.append(line(barx+cw,y-3,barx+cw,y+23,"#000",2))
            else:        # 앞 GPU, 뒤 CPU
                gw=barw*r['gn']/tot
                b.append(rect(barx,y,gw,20,GPU_COL))
                b.append(rect(barx+gw,y,barw-gw,20,CPU_COL))
                b.append(line(barx+gw,y-3,barx+gw,y+23,"#000",2))
            b.append(text(barx+barw+8,y+15,f"cpu {r['cpu']:.2f} / gpu {r['gpu']:.2f} ms",11,fill="#444"))
            y+=30
        y+=10
    save("fig1_cuts.svg", svg(W,H,"".join(b)))

# =========================================================================
# Fig 2: stage 시간 (cpu_stage, gpu_stage) 전 모델×방향 — 가로 막대(로그축)
# =========================================================================
import math
def fig_stages():
    rows=[]
    for m in models:
        for d,lab in (('cf','CPU-first'),('gf','GPU-first')):
            r=P[m][d]; rows.append((f"{SHORT[m]} {lab}", r['cpu'], r['gpu']))
    W,H=820,70+len(rows)*34; b=[]
    b.append(text(20,30,"[Fig 2] 방향별 stage 시간 (파랑=CPU-stage, 빨강=GPU-stage, ms)",16,weight="bold"))
    x0=170; xw=520; mx=max(max(c,g) for _,c,g in rows) or 1
    def sx(v): return xw*(v/mx)
    for gv in [1,5,10,20,40]:
        if gv<=mx:
            xx=x0+sx(gv); b.append(line(xx,50,xx,H-15,GRID)); b.append(text(xx,H-3,str(gv),10,"middle",fill="#999"))
    y=58
    for lab,c,g in rows:
        b.append(text(20,y+18,lab,12))
        b.append(rect(x0,y,sx(c),12,CPU_COL)); b.append(text(x0+sx(c)+4,y+11,f"{c:.2f}",10,fill="#444"))
        b.append(rect(x0,y+14,sx(g),12,GPU_COL)); b.append(text(x0+sx(g)+4,y+25,f"{g:.2f}",10,fill="#444"))
        y+=34
    save("fig2_stages.svg", svg(W,H,"".join(b)))

# =========================================================================
# Jackson 2기계 스케줄 시뮬 (Gantt 데이터)
# =========================================================================
def johnson(items,p1,p2):
    A=[x for x in items if p1(x)<=p2(x)]; B=[x for x in items if p1(x)>p2(x)]
    A.sort(key=p1); B.sort(key=p2,reverse=True); return A+B
def schedule(inst):
    CG=[x for x in inst if x['d']=='cf']; GC=[x for x in inst if x['d']=='gf']
    jcg=johnson(CG,lambda x:x['cpu'],lambda x:x['gpu'])
    jgc=johnson(GC,lambda x:x['gpu'],lambda x:x['cpu'])
    for x in inst: x['c']=None; x['g']=None
    t=0.0
    for x in jcg: x['c']=(t,t+x['cpu']); t+=x['cpu']
    cpu_free=t; t=0.0
    for x in jgc: x['g']=(t,t+x['gpu']); t+=x['gpu']
    gpu_free=t
    for x in jgc:
        s=max(cpu_free,x['g'][1]); x['c']=(s,s+x['cpu']); cpu_free=s+x['cpu']
    for x in jcg:
        s=max(gpu_free,x['c'][1]); x['g']=(s,s+x['gpu']); gpu_free=s+x['gpu']
    return max(cpu_free,gpu_free)

def build_inst(spec):
    inst=[]
    for tok in spec.split():
        a,c=tok.split(':'); m=ALIAS[a];
        mcf=max(P[m]['cf']['cpu'],P[m]['cf']['gpu']); mgf=max(P[m]['gf']['cpu'],P[m]['gf']['gpu'])
        d='gf' if mgf<mcf else 'cf'; r=P[m][d]
        for i in range(int(c)): inst.append({'m':a,'d':d,'cpu':r['cpu'],'gpu':r['gpu']})
    return inst

# =========================================================================
# Fig 3: 30-task 스케줄 순서 (예측 Gantt, 2레인)
# =========================================================================
def fig_gantt30():
    inst=build_inst("mob:7 yolo:7 lan:6 bise:5 wide:5")
    mk=schedule(inst)
    W,H=880,220; b=[]
    b.append(text(20,28,"[Fig 3] 30-task 예측 스케줄 (Jackson 순서, 파랑틀=CPU레인 빨강틀=GPU레인)",15,weight="bold"))
    b.append(text(20,48,f"색=모델(초록 mob, 파랑 lan, 빨강 bise, 주황 yolo, 보라 wide).  예측 makespan~={mk:.0f} ms",11,fill="#555"))
    x0=70; xw=780; y_cpu=80; y_gpu=140; lane_h=34
    def sx(v): return x0+xw*(v/mk)
    b.append(text(20,y_cpu+22,"CPU",12,weight="bold")); b.append(text(20,y_gpu+22,"GPU",12,weight="bold"))
    b.append(rect(x0,y_cpu,xw,lane_h,"#f4f8fb",CPU_COL)); b.append(rect(x0,y_gpu,xw,lane_h,"#fdf5f5",GPU_COL))
    for x in inst:
        col=MCOL[x['m']]
        if x['c']: s,e=x['c']; b.append(rect(sx(s),y_cpu+2,sx(e)-sx(s),lane_h-4,col,"#fff"))
        if x['g']: s,e=x['g']; b.append(rect(sx(s),y_gpu+2,sx(e)-sx(s),lane_h-4,col,"#fff"))
    for gv in range(0,int(mk)+1,50):
        b.append(line(sx(gv),70,sx(gv),y_gpu+lane_h+4,GRID)); b.append(text(sx(gv),y_gpu+lane_h+18,f"{gv}",10,"middle",fill="#999"))
    b.append(text(x0,205,"→ GPU 레인이 꽉 참(병목), CPU 레인은 거의 비어있음 = CPU가 느려 못 채움",11,fill="#a00"))
    save("fig3_gantt30.svg", svg(W,H,"".join(b)))

# =========================================================================
# Fig 4: 워크로드별 speedup (막대)
# =========================================================================
def fig_speedup():
    data=[("mob2+yolo2\n(4,mixed)",1.05),("each×2\n(10,mixed)",1.12),("mix\n(10,mixed)",1.01),
          ("each×3\n(15,mixed)",1.04),("mob×6\n(6,same)",0.90),("bise×4\n(4,same)",0.95)]
    W,H=760,360; b=[]
    b.append(text(20,30,"[Fig 4] 워크로드별 speedup (vs ALL-GPU)  — 1.0 위=이득, 아래=손해",15,weight="bold"))
    x0=60; y0=60; ph=230; bw=80; gap=30; mxv=1.2; mnv=0.8
    def sy(v): return y0+ph*(mxv-v)/(mxv-mnv)
    b.append(line(x0,sy(1.0),x0+len(data)*(bw+gap),sy(1.0),"#000",1.5,"4"))
    b.append(text(x0-8,sy(1.0)+4,"1.0",11,"end"))
    for gv in [0.8,0.9,1.0,1.1,1.2]:
        b.append(line(x0,sy(gv),x0+len(data)*(bw+gap),sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv:.1f}",10,"end",fill="#999"))
    x=x0+gap/2
    for lab,v in data:
        col="#59a14f" if v>=1.0 else "#e15759"
        top=sy(max(v,1.0)); bot=sy(min(v,1.0))
        b.append(rect(x,top,bw,bot-top,col))
        b.append(text(x+bw/2,top-6 if v>=1 else bot+16,f"{v:.2f}x",12,"middle",weight="bold",fill=col))
        for i,ln in enumerate(lab.split("\n")):
            b.append(text(x+bw/2,y0+ph+18+i*14,ln,10,"middle",fill="#444"))
        x+=bw+gap
    b.append(text(x0,H-10,"혼합=방향 섞임(이득), same=단일모델 버스트(손해). 근데 대부분 노이즈(±5~7%) 수준.",11,fill="#555"))
    save("fig4_speedup.svg", svg(W,H,"".join(b)))

# =========================================================================
# Fig 5: 30-task 비효율 (예측 vs 실측, 로그축)
# =========================================================================
def fig_ineff():
    # 예측(이상 Jackson) vs 실측 ALL-GPU / SCHEDULED
    inst=build_inst("mob:7 yolo:7 lan:6 bise:5 wide:5"); pred=schedule(inst)/1000.0  # ms→s
    data=[("예측(이상)\n%.2f s"%pred, pred, "#59a14f"),
          ("ALL-GPU\n실측 795 s", 795.13, "#e15759"),
          ("SCHEDULED\n실측 1070 s", 1070.04, "#a00000")]
    W,H=680,360; b=[]
    b.append(text(20,30,"[Fig 5] 30-task: 예측 vs 실측 (로그축) — GPU 컨텍스트 폭발로 ~2000배 비효율",15,weight="bold"))
    x0=90; y0=60; ph=210; bw=110; gap=60
    lo,hi=math.log10(0.1),math.log10(2000)
    def sy(v): return y0+ph*(hi-math.log10(max(v,0.1)))/(hi-lo)
    for gv in [0.1,1,10,100,1000]:
        b.append(line(x0,sy(gv),x0+len(data)*(bw+gap),sy(gv),GRID)); b.append(text(x0-8,sy(gv)+4,f"{gv:g}s",10,"end",fill="#999"))
    x=x0+gap/2
    for lab,v,col in data:
        top=sy(v); b.append(rect(x,top,bw,y0+ph-top,col))
        b.append(text(x+bw/2,top-6,f"{v:.0f}s" if v>=1 else f"{v:.2f}s",12,"middle",weight="bold",fill=col))
        for i,ln in enumerate(lab.split("\n")):
            b.append(text(x+bw/2,y0+ph+18+i*14,ln,11,"middle",fill="#444"))
        x+=bw+gap
    b.append(text(x0,H-14,"예측 %.1f초짜리 작업이 실측 795~1070초 = 컨텍스트30개+VRAM초과 thrashing."%pred,11,fill="#a00"))
    save("fig5_ineff.svg", svg(W,H,"".join(b)))

fig_cuts(); fig_stages(); fig_gantt30(); fig_speedup(); fig_ineff()
print("DONE ->", FIG)
