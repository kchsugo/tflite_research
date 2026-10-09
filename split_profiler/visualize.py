# -*- coding: utf-8 -*-
# split_profile.csv + 워크로드 → (1)모델 분할 (2)방향배정 (3)Jackson 큐 (4)Gantt 시각화
# 사용: python visualize.py "mob:2 yolo:2 lan:1 wide:1"
import sys, csv, os
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "split_profile.csv")

# 짧은 별칭
ALIAS = {"mob":"mobilenet_v2_full.tflite","lan":"lanenet_fp32.tflite",
         "bise":"BiseNet.tflite","yolo":"yolov4-tiny-416_ieie (1).tflite",
         "wide":"WideResNet50_float.tflite"}
SHORT = {v:k for k,v in ALIAS.items()}

# ---- CSV 로드 ----
prof = {}  # name -> {dir: {k,cpu,gpu,cn,gn}, 'Gwhole':..}
with open(CSV, newline='') as f:
    for r in csv.DictReader(f):
        nm=r['model']; d=r['direction']
        prof.setdefault(nm,{})
        row={'k':int(r['k']),'cpu':float(r['cpu_stage_ms']),'gpu':float(r['gpu_stage_ms']),
             'cn':int(r['cpu_nodes']),'gn':int(r['gpu_nodes'])}
        if d=='CPU-first': prof[nm]['cf']=row
        elif d=='GPU-first': prof[nm]['gf']=row
        elif d=='ALL-GPU': prof[nm]['Gwhole']=float(r['gpu_stage_ms'])

def bar(cn,gn,width=34,cchar='C',gchar='G'):
    tot=cn+gn
    if tot==0: return '?'*width
    c=round(cn/tot*width); return cchar*c + gchar*(width-c)

# ---- (1) 모델별 분할 ----
print("="*72); print(" (1) 모델별 분할  [C]=CPU 노드  [G]=GPU 노드   (stage 시간 ms)"); print("="*72)
for nm in prof:
    p=prof[nm]; s=SHORT.get(nm,nm[:8])
    for d,lab in (('cf','CPU-first'),('gf','GPU-first')):
        if d in p:
            r=p[d]
            # CPU-first: 앞이 CPU / GPU-first: 앞이 GPU
            if d=='cf': b=bar(r['cn'],r['gn'],cchar='C',gchar='G')
            else:       b=bar(r['gn'],r['cn'],cchar='G',gchar='C')  # 앞이 GPU
            print(f"  {s:5} {lab:9} k={r['k']:<4} |{b}|  cpu={r['cpu']:6.2f} gpu={r['gpu']:6.2f}")
    print()

# ---- 워크로드 파싱 ----
spec = sys.argv[1] if len(sys.argv)>1 else "mob:2 yolo:2 lan:1 wide:1"
inst=[]  # (name, dir 'cf'/'gf', cpu, gpu)
for tok in spec.split():
    a,c=tok.split(':'); nm=ALIAS.get(a,a); cnt=int(c)
    p=prof[nm]
    # per-task 최적 방향 = max(cpu,gpu) 작은 쪽
    mcf=max(p['cf']['cpu'],p['cf']['gpu']); mgf=max(p['gf']['cpu'],p['gf']['gpu'])
    d='gf' if mgf<mcf else 'cf'
    for i in range(cnt):
        inst.append([f"{SHORT.get(nm,nm[:4])}{i}", d, p[d]['cpu'], p[d]['gpu']])

# ---- (2) 방향 배정 ----
print("="*72); print(f" (2) 워크로드 & 방향 배정 (per-task 최적)   총 {len(inst)}개"); print("="*72)
CG=[x for x in inst if x[1]=='cf']  # CPU-first
GC=[x for x in inst if x[1]=='gf']  # GPU-first
for x in inst:
    dlab='CPU-first' if x[1]=='cf' else 'GPU-first'
    order = ('CPU→GPU' if x[1]=='cf' else 'GPU→CPU')
    print(f"  {x[0]:7} {dlab:9} ({order})  cpu={x[2]:6.2f} gpu={x[3]:6.2f}")
print(f"\n  → CPU-first(CG)={len(CG)}개,  GPU-first(GC)={len(GC)}개")

# ---- (3) Jackson (Johnson x2) ----
def johnson(items, p1, p2):  # p1,p2: lambda x -> time
    A=[x for x in items if p1(x)<=p2(x)]; B=[x for x in items if p1(x)>p2(x)]
    A.sort(key=p1); B.sort(key=p2, reverse=True); return A+B
jcg=johnson(CG, lambda x:x[2], lambda x:x[3])   # CG: 1기계=CPU(x[2]),2기계=GPU(x[3])
jgc=johnson(GC, lambda x:x[3], lambda x:x[2])   # GC: 1기계=GPU(x[3]),2기계=CPU(x[2])
cpu_q = jcg + jgc     # CPU 큐: CG먼저(그들의 CPU-part) → GC(그들의 CPU-part=2단계)
gpu_q = jgc + jcg     # GPU 큐: GC먼저 → CG
print("\n"+"="*72); print(" (3) Jackson 큐 (각 기계가 자기서 시작하는 것 먼저)"); print("="*72)
print("  CPU 큐:", " → ".join(x[0] for x in cpu_q))
print("  GPU 큐:", " → ".join(x[0] for x in gpu_q))

# ---- (4) 스케줄 시뮬레이션 (2기계 job shop, Jackson 순서) ----
# CG: stage1=CPU, stage2=GPU / GC: stage1=GPU, stage2=CPU
sched={}  # id -> {c:(s,e), g:(s,e)}
for x in inst: sched[x[0]]={}
# 초기 구간(의존 없음): CG의 CPU-part, GC의 GPU-part 는 t=0부터
cpu_free=0.0;
for x in jcg:  # CG CPU-part (stage1)
    s=cpu_free; e=s+x[2]; sched[x[0]]['c']=(s,e); cpu_free=e
gpu_free=0.0
for x in jgc:  # GC GPU-part (stage1)
    s=gpu_free; e=s+x[3]; sched[x[0]]['g']=(s,e); gpu_free=e
# 후속 구간: CG의 GPU-part(그 task CPU 끝난 뒤), GC의 CPU-part(그 task GPU 끝난 뒤)
for x in jgc:  # GC CPU-part (stage2) — GPU 끝난 뒤
    dep=sched[x[0]]['g'][1]; s=max(cpu_free,dep); e=s+x[2]; sched[x[0]]['c']=(s,e); cpu_free=e
for x in jcg:  # CG GPU-part (stage2) — CPU 끝난 뒤
    dep=sched[x[0]]['c'][1]; s=max(gpu_free,dep); e=s+x[3]; sched[x[0]]['g']=(s,e); gpu_free=e
makespan=max(cpu_free,gpu_free)

# ---- Gantt 렌더 ----
W=64
def lane(getseg):
    row=['.']*W
    for x in inst:
        seg=getseg(x)
        if not seg: continue
        s,e=seg
        a=int(s/makespan*W); b=max(a+1,int(e/makespan*W))
        ch=x[0][0]  # 모델 첫 글자
        for i in range(a,min(b,W)): row[i]=ch
    return ''.join(row)
print("\n"+"="*72); print(f" (4) 스케줄 Gantt (예측, stage 시간 기반)   makespan~={makespan:.2f} ms"); print("="*72)
print(f"  시간→ 0{'':^58}{makespan:.1f}ms")
print("  CPU |"+lane(lambda x: sched[x[0]].get('c'))+"|")
print("  GPU |"+lane(lambda x: sched[x[0]].get('g'))+"|")
print("       (글자=모델: m=mob l=lan b=bise y=yolo w=wide,  .=idle)")
# 겹침률
busy_c=sum(min(sched[x[0]]['c'][1],makespan)-sched[x[0]]['c'][0] for x in inst if 'c' in sched[x[0]])
busy_g=sum(min(sched[x[0]]['g'][1],makespan)-sched[x[0]]['g'][0] for x in inst if 'g' in sched[x[0]])
print(f"\n  CPU 가동률 {busy_c/makespan*100:4.0f}% , GPU 가동률 {busy_g/makespan*100:4.0f}%")
print(f"  (참고: ALL-GPU면 GPU가 모든 task를 직렬 → 더 김. 여기선 CPU-part가 겹쳐 단축)")
