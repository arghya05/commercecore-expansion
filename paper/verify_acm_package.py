"""Validate the stand-alone anonymous source archive on RunPod."""
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(args,cwd):
    return subprocess.run(args,cwd=cwd,check=True,capture_output=True,text=True).stdout
def main():
    if platform.system()!='Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('Run package verification on RunPod only.')
    manifest=json.loads((HERE/'acm/build_manifest.json').read_text())
    archive=HERE/'CommerceCore_Expansion_ACM_Source.zip'
    if sha(archive)!=manifest['source_zip_sha256']:raise ValueError('Source archive hash mismatch')
    report={'execution_location':'RunPod','source_zip_sha256':sha(archive),'outputs':{}}
    with tempfile.TemporaryDirectory(prefix='anonymous-acm-') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (tmp/item.filename).resolve().is_relative_to(tmp):raise ValueError('Unsafe ZIP entry')
            z.extractall(tmp)
        mapping={'commercecore_acm.tex':'paper.tex','commercecore_wsdm.tex':'paper_wsdm.tex','acm/BUILD.txt':'BUILD.txt'}
        for name,expected in manifest['sources_sha256'].items():
            original=HERE/name
            if sha(original)!=expected:raise ValueError('Source changed after build: '+name)
            want=original.read_bytes()
            if name=='commercecore_wsdm.tex':want=want.replace(b'commercecore_acm.tex',b'paper.tex')
            if (tmp/mapping.get(name,name)).read_bytes()!=want:raise ValueError('Archive source mismatch: '+name)
        for stem,public in [('paper','CommerceCore_Expansion_SIGIR_Draft'),('paper_wsdm','CommerceCore_Expansion_WSDM_Draft')]:
            for _ in range(3):run(['xelatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],tmp)
            log=(tmp/(stem+'.log')).read_text()
            if re.search(r'undefined references|undefined citations|(?:Reference|Citation) .+ undefined|Overfull \\[hv]box|Missing character',log,re.I):
                raise ValueError('Package compilation warning: '+stem)
            rebuilt=run(['pdftotext','-layout',str(tmp/(stem+'.pdf')),'-'],tmp)
            original=run(['pdftotext','-layout',str(HERE/(public+'.pdf')),'-'],tmp)
            if rebuilt.split()!=original.split():raise ValueError('Packaged PDF text differs: '+stem)
            pages=int(re.search(r'Pages:\s+(\d+)',run(['pdfinfo',str(tmp/(stem+'.pdf'))],tmp)).group(1))
            report['outputs'][public]={'text_matches_delivered_pdf':True,'pages':pages,'standalone_build':'PASS'}
    (HERE/'acm/package_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
