"""추적성 검사: 산문이 인용한 수치가 표/그림에 존재하는가.
verify_all.py 는 '표의 수치 == 원시 데이터'만 본다. 이 검사기는 반대 방향,
즉 '산문의 수치가 어디서 왔는가'를 본다. A1/A2/A3 유형이 여기서 잡힌다."""
import re,os
os.chdir("/home/jetson/jjyoo3/overleaf_work")
def walk(f,s=set(),u=set()):
    if not os.path.exists(f) or f in s: return u
    s.add(f)
    for l in open(f):
        for m in re.findall(r"\\(?:input|include)\{([^}]*)\}",l.split("%")[0]):
            q=(m if m.endswith(".tex") else m+".tex"); u.add(os.path.normpath(q)); walk(q,s,u)
    return u
F=sorted(walk("main-Jason.tex")|{"main-Jason.tex"})
S={f:re.sub(r"(?m)^\s*%.*$","",open(f).read()) for f in F}
ENV=r"table|table\*|longtable|tabular|tabular\*|tabularx"
POOL="".join("".join(re.findall(r"\\begin\{(?:"+ENV+r")\}.*?\\end\{(?:"+ENV+r")\}",s,re.S)) for s in S.values())
POOL+="".join("".join(re.findall(r"\\begin\{figure\}.*?\\end\{figure\}",s,re.S)) for s in S.values())
raw=set(re.findall(r"\d+(?:\.\d+)?",re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?"," ",POOL)))
VAR=set(raw)
for v in raw:
    try:
        x=float(v)
        for d in range(5): VAR.add(f"{x:.{d}f}")
    except ValueError: pass
def prose(s): return re.sub(r"\\begin\{(?:"+ENV+r"|figure)\}.*?\\end\{(?:"+ENV+r"|figure)\}","",s,flags=re.S)
bad=[]; nd=np=0
for f,s in S.items():
    p=prose(s)
    for m in re.finditer(r"(?<![\d.])\d+\.\d+(?![\d])",p):                 # 축 1: 소수
        v=m.group(0); nd+=1
        if re.match(r"(19|20)\d\d",v) or len(v.split('.')[1])<3 or v in VAR: continue
        bad.append(("소수",f.split('/')[-1],v," ".join(p[max(0,m.start()-70):m.start()+40].split())))
    for m in re.finditer(r"\$(\d{2,3})\\%\$",p):                            # 축 2: 두세 자리 퍼센트
        v=m.group(1); np+=1
        if v in VAR or any(abs(float(v)-float(x))<0.6 for x in raw if re.fullmatch(r"\d+(\.\d+)?",x)): continue
        bad.append(("퍼센트",f.split('/')[-1],v+"%"," ".join(p[max(0,m.start()-70):m.start()+30].split())))
print(f"  산문 소수 {nd}개 · 퍼센트 {np}개 검사")
print(f"  표/그림에 근거 없는 수치: {len(bad)}건")
for k,f,v,c in bad[:20]: print(f"   [{k}] {v:>10s} {f[:26]:28s} ...{c[-95:]}")
