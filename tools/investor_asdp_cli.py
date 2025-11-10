#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investor-ASDP Dev CLI (check → quick-fix → bump → changelog/readme → commit/tag)

Usage:
  python3 tools/investor_asdp_cli.py check [--json]
  python3 tools/investor_asdp_cli.py fix [--apply] [--json]
  python3 tools/investor_asdp_cli.py auto [--json]           # check→fix→recheck→bump→notes
  python3 tools/investor_asdp_cli.py release [--bump patch|minor|major] [--json]
"""

import argparse, json, os, re, subprocess, sys, shutil, time, difflib, glob, datetime
from pathlib import Path

# ---------- Basic utils ----------
def which(cmd): return shutil.which(cmd) is not None
def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, text=True, check=False)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
def read(p): return Path(p).read_text(encoding="utf-8", errors="ignore")
def write(p, s): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(s, encoding="utf-8")
def load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

ROOT = Path(".").resolve()
CFG_PATH = ROOT / "config" / "investor_asdp.dev.yaml"

# ---------- Check collectors (summarized) ----------
import xml.etree.ElementTree as ET
import logging

def clamp01(x):
    try: return max(0.0, min(1.0, float(x)))
except Exception:
return 0.0

def norm(value, minv, maxv, higher_is_better=True):
    if value is None: return 0.0
    try:
        v = float(value)
        if maxv == minv: return 0.0
        r = (v - minv) / (maxv - minv)
        return clamp01(r if higher_is_better else (1 - r))
except Exception:
return 0.0

def collect_coverage():
    cov = Path("coverage.xml")
    if not cov.exists(): return None
    try:
        root = ET.parse(str(cov)).getroot()
        rate = root.attrib.get("line-rate", None)
        if rate is not None: return clamp01(float(rate))
        # fallback: estimate
        lines_total = lines_covered = 0
        for n in root.iterfind(".//packages/package/classes/class"):
            lv = n.attrib.get("lines-valid") or n.attrib.get("lines")
            lc = n.attrib.get("lines-covered") or n.attrib.get("covered")
            if lv and lc: lines_total += int(float(lv)); lines_covered += int(float(lc))
        return clamp01((lines_covered / lines_total) if lines_total else 0.0)
except Exception:
return None

def collect_semgrep(t):
    if not which("semgrep"): return {"high":0,"med":0,"low":0,"skip":True}
    code,out,err = run(["semgrep","--json","--error","--quiet"], t)
    if code not in (0,1): return {"high":0,"med":0,"low":0,"err":err}
    try:
        data=json.loads(out or "{}"); c={"high":0,"med":0,"low":0}
        for r in data.get("results",[]):
            sev=((r.get("extra") or {}).get("severity") or "LOW").upper()
            if sev in ("CRITICAL","ERROR","HIGH"): c["high"]+=1
            elif sev=="MEDIUM": c["med"]+=1
            else: c["low"]+=1
        return c
except Exception:
return {"high":0,"med":0,"low":0}

def collect_bandit(t):
    if not which("bandit"): return {"high":0,"med":0,"low":0,"skip":True}
    code,out,err=run(["bandit","-r",".","-f","json","-q"], t)
    try:
        data=json.loads(out or "{}"); c={"high":0,"med":0,"low":0}
        for r in data.get("results",[]):
            sev=(r.get("issue_severity") or "LOW").upper()
            if sev=="HIGH": c["high"]+=1
            elif sev=="MEDIUM": c["med"]+=1
            else: c["low"]+=1
        return c
except Exception:
return {"high":0,"med":0,"low":0}

def collect_safety(t):
    if not which("safety"): return {"high":0,"med":0,"skip":True}
    has_reqs = any(Path(f).exists() for f in ("requirements.txt","pyproject.toml","poetry.lock","Pipfile.lock"))
    if not has_reqs: return {"high":0,"med":0,"skip":True}
    code,out,err=run(["safety","check","--json"], t)
    try:
        data=json.loads(out or "[]")
        high=sum(1 for v in data if (v.get("severity","") or "").upper() in ("HIGH","CRITICAL"))
        med=sum(1 for v in data if (v.get("severity","") or "").upper()=="MEDIUM")
        return {"high":high,"med":med}
except Exception:
return {"high":0,"med":0}

def collect_gitleaks(t):
    if not which("gitleaks"): return {"findings":0,"skip":True}
    code,out,err = run(["gitleaks","detect","--no-git","--report-format","json","--report-path","-"], t)
    if code not in (0,1): return {"findings":0}
    try:
        data=json.loads(out or "[]")
        return {"findings":len(data)}
except Exception:
return {"findings":0}

def collect_radon(t):
    if not which("radon"): return {"avg_cc":None,"skip":True}
    code,out,err=run(["radon","cc",".","-j"], t)
    try:
        data=json.loads(out or "{}"); total=cnt=0
        for files in data.values():
            for it in files:
                total+=it.get("complexity",0); cnt+=1
        return {"avg_cc": (total/cnt) if cnt else None}
except Exception:
return {"avg_cc":None}

def collect_jscpd(t):
    if not which("jscpd"): return {"dup_pct":None,"skip":True}
    code,out,err=run(["jscpd","--reporters","json","--silent"], t)
    try:
        j=Path("jscpd-report.json")
        if j.exists():
            data=json.loads(j.read_text(encoding="utf-8"))
            pct=float(((data.get("statistics") or {}).get("total") or {}).get("percentage",0))/100.0
            return {"dup_pct":pct}
except Exception:
pass
    return {"dup_pct":None}

def collect_trivy(t):
    if not which("trivy") or not Path("Dockerfile").exists(): return {"high":0,"skip":True}
    code,out,err=run(["trivy","fs","--format","json","."], t)
    try:
        data=json.loads(out or "{}"); high=0
        for r in data.get("Results",[]):
            for v in (r.get("Vulnerabilities") or []):
                if (v.get("Severity","") or "").upper() in ("HIGH","CRITICAL"): high+=1
        return {"high":high}
except Exception:
return {"high":0}

def spicy_scan(rules, top_n=5):
    findings=[]
    regexes=[(r["id"], re.compile(r["grep"]), r["advice"], r.get("autofix",False)) for r in rules]
    for root,_,files in os.walk("."):
        if ".git" in root.split(os.sep): continue
        for fn in files:
            if fn.endswith((".png",".jpg",".jpeg",".pdf",".lock",".min.js",".map")): continue
            p=Path(root)/fn
            try: text=p.read_text(encoding="utf-8", errors="ignore")
except Exception:
continue
            for rid,rx,ad,auto in regexes:
                for m in rx.finditer(text):
                    line=text.count("\n",0,m.start())+1
                    findings.append({"rule":rid,"file":str(p),"line":line,"advice":ad,"autofix":auto})
    # dedup by file+rule
    uniq={}
    for f in findings:
        key=(f["rule"],f["file"])
        if key not in uniq: uniq[key]=f
    return list(uniq.values())[:top_n]

def score(cfg, M):
    # Integrity
    secrets_ok = 1.0 if M["gitleaks"]["findings"]==0 else 0.0
    high_total = M["semgrep"]["high"] + M["bandit"]["high"] + M["trivy"]["high"] + M["safety"]["high"]
    sast_ok = 1.0 if high_total==0 else 0.0
    integrity = (secrets_ok + sast_ok)/2.0
    # Resonance
    coverage = M["coverage"] if M["coverage"] is not None else 0.0
    resonance = coverage
    # Stability
    cc = M["radon"]["avg_cc"]
    cc_norm = 1.0 if cc is None else norm(cc, 1, cfg["metrics"]["thresholds"]["complexity_avg_max"], higher_is_better=False)
    dup = M["jscpd"]["dup_pct"]
    dup_norm = 1.0 if dup is None else norm(dup, 0.0, cfg["metrics"]["thresholds"]["duplication_pct_max"], higher_is_better=False)
    stability = (cc_norm + dup_norm)/2.0
    w=cfg["weights"]
    omega = w["integrity"]*integrity + w["resonance"]*resonance + w["stability"]*stability
    return clamp01(omega), integrity, resonance, stability

def run_checks(cfg):
    T=cfg["tools"]["timeouts_sec"]
    M={}
    M["coverage"]=collect_coverage()
    M["semgrep"]=collect_semgrep(T["semgrep"])
    M["bandit"]=collect_bandit(T["bandit"])
    M["safety"]=collect_safety(T["safety"])
    M["gitleaks"]=collect_gitleaks(T["gitleaks"])
    M["radon"]=collect_radon(T["radon"])
    M["jscpd"]=collect_jscpd(T["jscpd"])
    M["trivy"]=collect_trivy(T["trivy"])
    omega, integ, reso, stab = score(cfg, M)
    gates=cfg["gates"]
    status = "Certified" if omega>=gates["certified"] else ("Review" if omega>=gates["review"] else "Fail")
    return {
        "omega": round(omega,4), "status": status,
        "attributes": {"integrity":round(integ,4),"resonance":round(reso,4),"stability":round(stab,4)},
        "metrics": M
    }

# ---------- Auto-fixers ----------
def patch_preview(old, new, file):
    diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=file, tofile=file)
    return "".join(diff)

def ensure_logger_header(text):
    lines=text.splitlines()
    head="\n".join(lines[:20])
    needs_import = "import logging" not in head
    needs_logger = "getLogger(__name__)" not in text
    ins=[]
    if needs_import: ins.append("import logging")
    if needs_logger: ins.append("logger = logging.getLogger(__name__)")
    if not ins: return text
    # after first import block
    for i,l in enumerate(lines[:50]):
        if l.strip().startswith("import ") or l.strip().startswith("from "):
            last=i
    try:
        last
        lines.insert(last+1, "\n".join(ins))
    except NameError:
        lines = ins + [""] + lines
    return "\n".join(lines)

def fix_print_to_logger(text):
    # naive replace: logger.info("x") → logger.info("x")
    # keep f-strings/format as is
    new = re.sub(r'(^|\s)print\(', r'\1logger.info(', text)
    if new != text:
        new = ensure_logger_header(new)
    return new

def fix_broad_except(text):
    return re.sub(r'(?m)^\s*except\s*:\s*', 'except Exception:\n', text)

def fix_requests_timeout(text):
    # Insert ", timeout=10" if requests.*( ... ) lacks timeout=
    def repl(m):
        call=m.group(0)
        if "timeout=" in call: return call
        # handle trailing ) possibly followed by newline or whitespace
        return re.sub(r'\)\s*', ', timeout=10)', call, count=1)
    pattern=r'requests\.(get|post|put|patch|delete)\([^)]*\)'
    return re.sub(pattern, repl, text)

def apply_autofixes(cfg, findings):
    # only Python files for programmatic fixes to avoid false positives
    applied=[]
    for f in findings:
        if not f["autofix"]: continue
        p=Path(f["file"])
        if not p.suffix.lower() in (".py",): continue
        try:
            old=read(p)
            new=old
            if f["rule"]=="print_in_lib":
                new=fix_print_to_logger(new)
            elif f["rule"]=="broad_except":
                new=fix_broad_except(new)
            elif f["rule"]=="requests_no_timeout":
                new=fix_requests_timeout(new)
            if new!=old:
                write(p, new)
                applied.append({"file":str(p), "rule":f["rule"], "diff":patch_preview(old,new,str(p))})
        except Exception as e:
            applied.append({"file":str(p), "rule":f["rule"], "error":str(e)})
    return applied

# ---------- Version bump & notes ----------
def find_version_targets(cfg):
    targets=[]
    for spec in cfg["version_files"]:
        kind=spec["kind"]; path=spec["path"]
        if "**" in path:
            for p in glob.glob(path, recursive=True):
                if Path(p).is_file(): targets.append({"kind":kind,"path":p})
        else:
            if Path(path).exists(): targets.append({"kind":kind,"path":path})
    return targets

def parse_version_from(content, kind):
    if kind=="pyproject":
        m=re.search(r'(?m)^\s*version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)["\']', content)
        return m.group(1) if m else None
    if kind=="package":
        m=re.search(r'(?m)"version"\s*:\s*"(.*?)"', content)
        return m.group(1) if m else None
    if kind=="setupcfg":
        m=re.search(r'(?m)^\s*version\s*=\s*([0-9]+\.[0-9]+\.[0-9]+)', content)
        return m.group(1) if m else None
    if kind=="initpy":
        m=re.search(r'__version__\s*=\s*["\'](.*?)["\']', content)
        return m.group(1) if m else None
    if kind=="plain":
        m=re.search(r'([0-9]+\.[0-9]+\.[0-9]+)', content)
        return m.group(1) if m else content.strip()
    return None

def bump_semver(ver, bump):
    major,minor,patch=[int(x) for x in ver.split(".")]
    if bump=="major": major+=1; minor=0; patch=0
    elif bump=="minor": minor+=1; patch=0
    else: patch+=1
    return f"{major}.{minor}.{patch}"

def write_version(content, kind, new_ver):
    if kind=="pyproject":
        return re.sub(r'(?m)^(\s*version\s*=\s*["\'])[0-9]+\.[0-9]+\.[0-9]+(["\'])', rf'\g<1>{new_ver}\2', content)
    if kind=="package":
        return re.sub(r'(?m)("version"\s*:\s*")[^"]+(")', rf'\1{new_ver}\2', content)
    if kind=="setupcfg":
        return re.sub(r'(?m)^(\s*version\s*=\s*)[0-9]+\.[0-9]+\.[0-9]+', rf'\1{new_ver}', content)
    if kind=="initpy":
        if re.search(r'__version__', content):
            return re.sub(r'(?m)(__version__\s*=\s*["\']).*(["\'])', rf'\1{new_ver}\2', content)
        else:
            return f'__version__ = "{new_ver}"\n{content}'
    if kind=="plain":
        return f"{new_ver}\n"
    return content

def decide_bump(cfg, before, after, spicy_applied):
    # 기본: patch. 큰 개선(Ω +0.1 이상)이나 SAST High→0 전환 시 minor, API 영향 있으면 major(수동)
    bump="patch"
    if (after["omega"] - before["omega"]) >= 0.10: bump="minor"
    if before["metrics"]["semgrep"]["high"]>0 or before["metrics"]["bandit"]["high"]>0:
        if after["metrics"]["semgrep"]["high"]==0 and after["metrics"]["bandit"]["high"]==0:
            bump="minor"
    # config override
    s= (load_yaml(CFG_PATH)).get("modes",{}).get("bump_strategy","auto")
    if s in ("patch","minor","major"): bump = s
    return bump

def update_changelog(cfg, new_ver, before, after, fix_applied):
    path=cfg["changelog"]["file"]
    date=datetime.date.today().isoformat()
    heading=cfg["changelog"]["heading"].replace("${version}",new_ver).replace("${date}",date)
    coverage_p=int(round((after["metrics"]["coverage"] or 0)*100))
    cc_avg = after["metrics"]["radon"].get("avg_cc")
    dup_p = int(round(((after["metrics"]["jscpd"].get("dup_pct") or 0)*100)))
    bullets=[b.replace("${fix_count}",str(len(fix_applied)))\
               .replace("${sast_high}",str(after["metrics"]["semgrep"].get("high",0)+after["metrics"]["bandit"].get("high",0)))\
               .replace("${secrets}",str(after["metrics"]["gitleaks"].get("findings",0)))\
               .replace("${vulns_high}",str(after["metrics"]["trivy"].get("high",0)))\
               .replace("${coverage_p}",str(coverage_p))\
               .replace("${cc_avg}", f"{cc_avg:.2f}" if isinstance(cc_avg,(int,float)) else "n/a")\
               .replace("${dup_p}",str(dup_p))
             for b in cfg["changelog"]["bullets"]]
    body = heading + "\n" + "\n".join([f"- {b}" for b in bullets]) + "\n\n"
    existing = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    write(path, body + existing)

def update_readme_status(cfg, after):
    block_start=cfg["readme"]["status_block"]["start"]
    block_end=cfg["readme"]["status_block"]["end"]
    title=cfg["readme"]["status_block"]["title"]
    readme="README.md"
    content = Path(readme).read_text(encoding="utf-8") if Path(readme).exists() else f"# {title}\n\n"
    status = f"{block_start}\n### {title}\n" \
             f"- Ω: **{after['omega']}** ({after['status']})\n" \
             f"- Integrity/Resonance/Stability: {after['attributes']['integrity']}/{after['attributes']['resonance']}/{after['attributes']['stability']}\n" \
             f"- Coverage: {after['metrics']['coverage']}\n" \
             f"- SAST High: {after['metrics']['semgrep'].get('high',0)+after['metrics']['bandit'].get('high',0)} | Secrets: {after['metrics']['gitleaks'].get('findings',0)} | Vulns High: {after['metrics']['trivy'].get('high',0)}\n" \
             f"{block_end}\n"
    if block_start in content and block_end in content:
        content = re.sub(rf'{re.escape(block_start)}.*?{re.escape(block_end)}', status, content, flags=re.S)
    else:
        content += "\n" + status + "\n"
    write(readme, content)

def git_commit_tag(cfg, new_ver):
    if not which("git"): return
    run(["git","add","-A"])
    run(["git","commit","-m", f"chore(asdp): auto-fix + release v{new_ver}"])
    if cfg["modes"].get("create_tag", True):
        run(["git","tag", f"{cfg['changelog'].get('tag_prefix','v')}{new_ver}"])

# ---------- Commands ----------
def cmd_check(args):
    cfg=load_yaml(CFG_PATH)
    res=run_checks(cfg)
    spicy=spicy_scan(cfg.get("spicy_rules",[]), top_n=5)
    out={"result":res, "spicy_top":spicy}
    if args.json: logger.info(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        logger.info("\n== Investor-ASDP :: CHECK ==")
        logger.info(f"Ω: {res['omega']} → {res['status']}")
        logger.info("Attributes:", res["attributes"])
        logger.info("Metrics: SAST High=", res["metrics"]["semgrep"].get("high",0)+res["metrics"]["bandit"].get("high",0),
              " Secrets=",res["metrics"]["gitleaks"].get("findings",0),
              " VulnsHigh=",res["metrics"]["trivy"].get("high",0))
        if spicy:
            logger.info("\nTop Quick Fix:")
            for i,f in enumerate(spicy,1):
                logger.info(f" {i}. [{f['rule']}] {f['file']}:{f['line']} → {f['advice']} (auto={f['autofix']})")
    return 0 if res["status"]=="Certified" else (2 if res["status"]=="Review" else 3)

def cmd_fix(args):
    cfg=load_yaml(CFG_PATH)
    spicy=spicy_scan(cfg.get("spicy_rules",[]), top_n=50)  # 넉넉히 모아둠
    candidates=[f for f in spicy if f["autofix"]]
    if not candidates:
        out={"applied":[], "note":"no autofix candidates"}
        return logger.info(json.dumps(out, ensure_ascii=False, indent=2)) if args.json else logger.info("No autofix candidates"), 0
    applied = apply_autofixes(cfg, candidates)
    if args.json: logger.info(json.dumps({"applied":applied}, ensure_ascii=False, indent=2))
    else:
        logger.info("\n== Investor-ASDP :: FIX ==")
        for a in applied:
            if "diff" in a:
                logger.info(f"\n# {a['file']} ({a['rule']})\n" + a["diff"])
            else:
                logger.info(f"- {a['file']} ({a['rule']}) -> {a.get('error','applied')}")
    return 0

def cmd_auto(args):
    cfg=load_yaml(CFG_PATH)
    before=run_checks(cfg)
    spicy_before=spicy_scan(cfg.get("spicy_rules",[]), top_n=50)
    applied=apply_autofixes(cfg, [f for f in spicy_before if f["autofix"]])
    after=run_checks(cfg)
    bump = decide_bump(cfg, before, after, applied)
    # find current version
    targets=find_version_targets(cfg)
    new_ver=None; changed=[]
    for t in targets:
        p=Path(t["path"])
        if not p.exists(): continue
        content=read(p)
        cur=parse_version_from(content, t["kind"]) or "0.0.0"
        nv=bump_semver(cur, bump)
        if nv!=cur:
            write(p, write_version(content, t["kind"], nv))
            new_ver=nv; changed.append(str(p))
            break
    # notes
    if new_ver:
        update_changelog(cfg, new_ver, before, after, applied)
        update_readme_status({"readme":{"status_block":cfg["readme"]["status_block"]}}, after)
        git_commit_tag(cfg, new_ver)
    out={"before":before,"after":after,"bump":bump,"version":new_ver,"fix_applied":applied}
    if args.json: logger.info(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        logger.info("\n== Investor-ASDP :: AUTO ==")
        logger.info(f"Ω before→after: {before['omega']} → {after['omega']}  | status: {after['status']}")
        logger.info(f"bump: {bump}  new version: {new_ver or '(no version file found)'}")
        logger.info(f"applied fixes: {len(applied)}")
    return 0 if after["status"]=="Certified" else (2 if after["status"]=="Review" else 3)

def cmd_release(args):
    cfg=load_yaml(CFG_PATH)
    before=run_checks(cfg)
    applied=apply_autofixes(cfg, [f for f in spicy_scan(cfg.get("spicy_rules",[]), top_n=50) if f["autofix"]])
    after=run_checks(cfg)
    bump = args.bump or decide_bump(cfg, before, after, applied)
    targets=find_version_targets(cfg); new_ver=None
    for t in targets:
        p=Path(t["path"])
        if not p.exists(): continue
        c=read(p); cur=parse_version_from(c, t["kind"]) or "0.0.0"
        nv=bump_semver(cur, bump)
        if nv!=cur:
            write(p, write_version(c, t["kind"], nv))
            new_ver=nv; break
    if new_ver:
        update_changelog(cfg, new_ver, before, after, applied)
        update_readme_status({"readme":{"status_block":cfg["readme"]["status_block"]}}, after)
        if (load_yaml(CFG_PATH)).get("modes",{}).get("commit",True): git_commit_tag(cfg, new_ver)
    out={"before":before,"after":after,"bump":bump,"version":new_ver,"fix_applied":applied}
    if args.json: logger.info(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        logger.info("\n== Investor-ASDP :: RELEASE ==")
        logger.info(f"Ω: {before['omega']} → {after['omega']}  | bump={bump} → v{new_ver or '?'}")
    return 0

def main():
    ap=argparse.ArgumentParser(prog="asdp")
    sub=ap.add_subparsers(dest="cmd")

    p1=sub.add_parser("check"); p1.add_argument("--json", action="store_true")
    p2=sub.add_parser("fix");   p2.add_argument("--apply", action="store_true"); p2.add_argument("--json", action="store_true")
    p3=sub.add_parser("auto");  p3.add_argument("--json", action="store_true")
    p4=sub.add_parser("release"); p4.add_argument("--bump", choices=["patch","minor","major"]); p4.add_argument("--json", action="store_true")

    args=ap.parse_args()
    if not CFG_PATH.exists():
        logger.info(f"[ERR] config not found: {CFG_PATH}", file=sys.stderr); return 1

    if args.cmd=="check": return cmd_check(args)
    if args.cmd=="fix":   return cmd_fix(args)
    if args.cmd=="auto":  return cmd_auto(args)
    if args.cmd=="release": return cmd_release(args)

    ap.print_help(); return 0

if __name__=="__main__":
    sys.exit(main())