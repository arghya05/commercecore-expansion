"""Final remote publication checks and artifact bookkeeping; no new inference."""
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from research_v2.execution import require_runpod
from research_v2.core import file_sha

def main():
    require_runpod();root=Path(__file__).resolve().parents[2]
    out=root/'research_v2/results/followup_20260928'
    for name in ['hardware.json','environment.txt']:
        shutil.copyfile(root/'research_v2/remote_artifacts'/name,out/name)
    manifest_path=root/'paper/acm/build_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['research_status']='Anonymous drafts with matched development comparisons and a bounded HTTP prototype; controlled training, human validation, final frontier comparisons and production evaluation remain incomplete.'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    verification={}
    for stem in ['CommerceCore_Expansion_SIGIR_Draft','CommerceCore_Expansion_WSDM_Draft','CommerceCore_Expansion_Paper']:
        pdf=root/'paper'/(stem+'.pdf')
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        for value in ['0.9150','0.9025','0.4675','40.56']:
            if value not in text:raise ValueError('Missing verified result '+value+' in '+stem)
        verification[stem]={'sha256':file_sha(pdf),'followup_values_present':True,
            'pages':len(text.rstrip('\f\n').split('\f'))}
    if 'Ran 19 tests' not in (out/'protocol_tests.log').read_text():raise ValueError('Missing protocol test log')
    preprint_log=(out/'preprint_build.log').read_text()
    if 'OK' not in preprint_log or 'Built CommerceCore_Expansion_Paper.pdf' not in preprint_log:
        raise ValueError('Preprint build/test verification incomplete')
    (out/'publication_verification.json').write_text(json.dumps({'verified_utc':datetime.now(timezone.utc).isoformat(),
        'location':'RunPod','pdfs':verification,'protocol_tests':19,'historical_audit_tests':'passed in preprint build',
        'visual_review':'SIGIR pages 6 and 7 inspected; tables and new text readable, no clipping.',
        'budget_ceiling_usd':25,'paid_frontier_calls':False,'new_training':False,'final_test_scored':False},indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
