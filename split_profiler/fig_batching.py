# -*- coding: utf-8 -*-
# 배칭 실험 시각화 → docs/figures/figB*.svg
import os,sys,math
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"docs","figures"); os.makedirs(FIG,exist_ok=True)
C={"mob":"#2E7D32","yolo":"#2E5FA0","lane":"#7B4EA8","wide":"#E8820C","bise":"#C00000"}
TXT="#222";GRID="#ddd";REF="#9AA0A6"
def tx(x,y,s,sz=13,a="start",f=TXT,w="normal"):return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Arial" font-size="{sz}" text-anchor="{a}" fill="{f}" font-weight="{w}">{s}</text>'
def ln(x1,y1,x2,y2,st=GRID,w=1,d=""):
    dd=f' stroke-dasharray="{d}"' if d else "";return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{st}" stroke-width="{w}"{dd}/>'
def cr(x,y,r,f):return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{f}"/>'
def rc(x,y,w,h,f,op=1):return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" fill="{f}" opacity="{op}"/>'
def svg(w,h,b):return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>{b}</svg>'
def save(n,s): open(os.path.join(FIG,n),"w",encoding="utf-8").write(s);print("wrote",n)

B=[1,2,4,8,16,32]
NAME={"mob":"mobilenet","yolo":"yolov4-tiny","lane":"lanenet","wide":"WideResNet50","bise":"BiseNet"}
PT={ # per-task(ms) by B  (None=미지원)
 "mob":[1.246,0.670,0.571,0.376,0.318,0.292],
 "yolo":[2.912,1.964,1.635,1.525,1.450,1.406],
 "lane":[3.345,2.852,2.607,2.614,2.479,None],
 "wide":[7.351,5.934,4.548,3.861,3.778,3.698],
 "bise":[16.323,15.636,15.140,15.071,15.729,None],
}
SP={k:PT[k][0]/min(v for v in PT[k] if v) for k,v in PT.items()}  # max speedup

# ---- figB1: per-task vs batch (로그 x, 로그 y) ----
def figB1():
    W,H=780,440;b=[tx(20,28,"[B1] batch 크기별 per-task 시간 (ms, 양축 로그)",15,w="bold")]
    b.append(tx(20,47,"선이 많이 내려갈수록 배칭 이득 큼. 작은 모델(mobilenet)이 가장 가파름.",11,f="#555"))
    x0=70;y0=72;pw=600;ph=300
    def sx(bb):return x0+pw*(math.log2(bb)/math.log2(32))
    ymin,ymax=0.25,20.0
    def sy(v):return y0+ph*(math.log10(v/ymin)/math.log10(ymax/ymin))*-1+ph
    # grid y
    for gv in [0.25,0.5,1,2,4,8,16]:
        b.append(ln(x0,sy(gv),x0+pw,sy(gv),GRID));b.append(tx(x0-6,sy(gv)+4,str(gv),10,"end",f="#999"))
    for bb in B: b.append(tx(sx(bb),y0+ph+16,str(bb),10,"middle"))
    b.append(tx(x0+pw/2,y0+ph+34,"batch 크기 B (log)",11,"middle",f="#555"));b.append(tx(20,y0-6,"per-task(ms)",10,f="#999"))
    for k,ys in PT.items():
        pts=[(sx(B[i]),sy(ys[i])) for i in range(len(B)) if ys[i]]
        for i in range(len(pts)-1): b.append(ln(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],C[k],2.5))
        for p in pts: b.append(cr(p[0],p[1],3.5,C[k]))
        b.append(tx(pts[-1][0]+6,pts[-1][1]+4,f"{NAME[k]} ({SP[k]:.1f}x)",10,f=C[k],w="bold"))
    save("figB1_pertask.svg",svg(W,H,"".join(b)))

# ---- figB2: 모델별 최대 speedup ----
def figB2():
    W,H=620,360;b=[tx(20,28,"[B2] 배칭 최대 speedup (per-task, vs batch=1)",15,w="bold")]
    order=["mob","yolo","wide","lane","bise"]
    x0=90;y0=65;ph=230;slot=(W-x0-30)/len(order);bw=slot*0.5;mx=4.5
    def sy(v):return y0+ph*(1-v/mx)
    for gv in [1,2,3,4]:
        b.append(ln(x0,sy(gv),W-30,sy(gv),GRID if gv!=1 else "#aaa"));b.append(tx(x0-6,sy(gv)+4,f"{gv}x",10,"end",f="#999"))
    for i,k in enumerate(order):
        cx=x0+slot*i+slot/2;sp=SP[k]
        b.append(rc(cx-bw/2,sy(sp),bw,y0+ph-sy(sp),C[k]))
        b.append(tx(cx,sy(sp)-6,f"{sp:.2f}x",12,"middle",w="bold",f=C[k]))
        b.append(tx(cx,y0+ph+18,NAME[k],10,"middle"))
    b.append(tx(x0,H-12,"작은/저활용 모델일수록 배칭 이득 큼 · 무거운 모델(bise)은 이미 GPU 포화",10,f="#555"))
    save("figB2_speedup.svg",svg(W,H,"".join(b)))

# ---- figB3: 버스트 makespan 투영 (N=50, batch 없음 vs batch=16) ----
def figB3():
    N=50;Bb=16;bi=B.index(Bb)
    order=["mob","yolo","wide"]
    W,H=640,380;b=[tx(20,28,f"[B3] 같은 모델 {N}개 버스트 makespan 투영 (ms)",15,w="bold")]
    b.append(tx(20,47,f"하나씩(batch=1) vs 배칭(batch={Bb}). per-task×{N} 투영. GPU 저활용 모델에서 극적.",11,f="#555"))
    x0=80;y0=75;ph=240;slot=(W-x0-30)/len(order);bw=slot*0.3
    vals={k:(PT[k][0]*N, PT[k][bi]*N) for k in order}
    mx=max(v[0] for v in vals.values())*1.1
    def sy(v):return y0+ph*(1-v/mx)
    for gv in range(0,int(mx)+1,50):
        b.append(ln(x0,sy(gv),W-30,sy(gv),GRID));b.append(tx(x0-6,sy(gv)+4,str(gv),10,"end",f="#999"))
    b.append(tx(20,y0-4,"ms",10,f="#999"))
    for i,k in enumerate(order):
        cx=x0+slot*i+slot/2;nb,wb=vals[k]
        b.append(rc(cx-bw-2,sy(nb),bw,y0+ph-sy(nb),REF));b.append(tx(cx-bw/2-2,sy(nb)-4,f"{nb:.0f}",9,"middle",f="#777"))
        b.append(rc(cx+2,sy(wb),bw,y0+ph-sy(wb),C[k]));b.append(tx(cx+bw/2+2,sy(wb)-4,f"{wb:.0f}",9,"middle",f=C[k]))
        b.append(tx(cx,sy(wb)-20,f"{nb/wb:.1f}x",12,"middle",w="bold",f=C[k]))
        b.append(tx(cx,y0+ph+18,NAME[k],11,"middle"))
    lx=x0;ly=H-14
    b.append(rc(lx,ly-10,12,12,REF));b.append(tx(lx+16,ly,"batch=1 (하나씩)",11))
    b.append(rc(lx+180,ly-10,12,12,C["mob"]));b.append(tx(lx+196,ly,"batch=16",11))
    save("figB3_burst.svg",svg(W,H,"".join(b)))

if __name__=="__main__":
    figB1();figB2();figB3();print("done")
