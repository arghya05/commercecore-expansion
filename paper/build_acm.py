"""Build and validate both anonymous ACM drafts on RunPod only."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def run(command,cwd=HERE):
    return subprocess.run(command,cwd=cwd,check=True,text=True,capture_output=True).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if platform.system()!='Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('Compile and validate this paper on RunPod only.')
    compiler=shutil.which('xelatex')
    if not compiler:raise SystemExit('RunPod requires XeLaTeX and Poppler.')
    build=HERE/'acm/build';build.mkdir(exist_ok=True)
    audit_log=run([sys.executable,'audit_evidence.py'])
    (build/'evidence_audit.log').write_text(audit_log)
    runs={}
    for stem,public in [('commercecore_acm','CommerceCore_Expansion_SIGIR_Draft'),
                        ('commercecore_wsdm','CommerceCore_Expansion_WSDM_Draft')]:
        command=[compiler,'-interaction=nonstopmode','-halt-on-error','-file-line-error',
                 '-output-directory='+str(build),stem+'.tex']
        for _ in range(3):
            result=subprocess.run(command,cwd=HERE,text=True,capture_output=True)
            (build/(stem+'.compiler.log')).write_text(result.stdout+result.stderr)
            if result.returncode:raise SystemExit('LaTeX failed; inspect '+str(build/(stem+'.compiler.log')))
        log=(build/(stem+'.log')).read_text()
        problems=re.findall(r'^.*(?:undefined references|undefined citations|(?:Reference|Citation) .+ undefined|Overfull \\[hv]box|Missing character).*$',log,re.M|re.I)
        if problems:raise SystemExit('\n'.join(problems))
        pdf=build/(stem+'.pdf');text=run(['pdftotext','-layout',str(pdf),'-'])
        info=run(['pdfinfo',str(pdf)]);fonts=run(['pdffonts',str(pdf)])
        (build/(stem+'.txt')).write_text(text)
        (build/(stem+'.pdfinfo.txt')).write_text(info)
        (build/(stem+'.fonts.txt')).write_text(fonts)
        for identifying in ['arghya','commercecore','github.com/arghya','@gmail.com']:
            if identifying in (text+'\n'+info).lower():raise SystemExit('Anonymity check failed: '+identifying)
        aux=(build/(stem+'.aux')).read_text()
        page=re.search(r'\\newlabel\{end:main\}\{\{[^}]*\}\{(\d+)\}',aux)
        if not page or int(page.group(1))>9:raise SystemExit('Main content page limit failed')
        total=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
        font_lines=fonts.splitlines()[2:]
        if any(not re.search(r'\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line) for line in font_lines if line.strip()):
            raise SystemExit('A font may not be embedded; inspect pdffonts output')
        dest=HERE/(public+'.pdf');shutil.copyfile(pdf,dest)
        runs[public]={'main_content_end_page':int(page.group(1)),'total_pages':total,
                     'pdf_sha256':sha(dest),'fonts_embedded':True,'anonymity_text_metadata_check':'PASS'}
    sources=['commercecore_acm.tex','commercecore_wsdm.tex','acmart.cls','acm/body.tex','acm/references.tex','acm/dev_rows.tex',
             'acm/vendor/acmart.dtx','acm/vendor/acmart.ins','acm/vendor/README',
             'acm/BUILD.txt','evidence/numbers.tex','evidence/mixture_rows.tex','evidence/relevance_class_rows.tex','evidence/functional_sweep.csv']
    archive=HERE/'CommerceCore_Expansion_ACM_Source.zip'
    # Generic entry-point names avoid embedding the public project's name in review source files.
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            target={'commercecore_acm.tex':'paper.tex','commercecore_wsdm.tex':'paper_wsdm.tex','acm/BUILD.txt':'BUILD.txt'}.get(name,name)
            content=(HERE/name).read_bytes()
            if name=='commercecore_wsdm.tex':content=content.replace(b'commercecore_acm.tex',b'paper.tex')
            entry=zipfile.ZipInfo(target,date_time=(2026,9,28,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,content)
    manifest={'compiler':run([compiler,'--version']).splitlines()[0],
              'execution_location':'RunPod','pod_id':os.environ['RUNPOD_POD_ID'],
              'sources_sha256':{name:sha(HERE/name) for name in sources},
              'outputs':runs,'source_zip_sha256':sha(archive),
              'research_status':'Anonymous drafts; new matched comparisons, controlled ablations, human validation and serving measurements remain incomplete.',
              'submission_status':'NOT_SUBMITTED; WSDM 2027 deadline has passed; select an eligible future cycle.'}
    (HERE/'acm/build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
