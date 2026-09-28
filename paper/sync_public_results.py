"""Generate synchronized public summaries from the audited evidence on RunPod."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GITHUB='https://github.com/arghya05/commercecore-expansion'
PILOT='research_v2/results/runpod_pilot_20260928/qwen_base_gpu_dev_v2'
SLUGS=['understand-v1','match-v1','functional-relation-v1','relevance-singletoken-v1']
def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def outputs():
    audit=read('paper/evidence/audit_results.json')
    pilot=read(PILOT+'/report.json')
    manifest=read(PILOT+'/manifest.json')
    if sha(PILOT+'/manifest.json')!=pilot['manifest_sha256']:raise ValueError('Pilot manifest mismatch')
    if sha(PILOT+'/predictions.jsonl')!=pilot['predictions_sha256']:raise ValueError('Pilot output mismatch')
    if manifest['phase']!='development' or manifest['test_accessed']:raise ValueError('Unexpected pilot scope')
    m=audit['macros'];d=pilot['metrics'];f=audit['functional_metrics'];q=audit['single_token_metrics']
    shared=audit['results']['shared_v1_rescore']
    common=f"""## Audited results — 2026-09-28

The historical adapter results below use the original scoring rules and local
evaluation subsets. They are not official leaderboard results. Dataset, metric
and exposure qualifications are part of each result.

| Model / task | Recorded result | Evaluation and interpretation |
|---|---|---|
| Extraction specialist | Brand F1 **{m['ExtractionBrand']}**; color F1 **{m['ExtractionColor']}** | 200 rows; permissive field matching; 2 evaluation texts overlap training |
| Shared adapter: relevance | Accuracy **{m['SharedRelevance']}** | 1,000 rows; logged v1 rescore |
| Shared adapter: identity | Accuracy **{m['SharedIdentity']}** | 3,500 rows; logged v1 rescore; legacy substring scorer |
| Shared adapter: functional relation | Accuracy **{shared['match_functional_relation']:.3f}** (9/13) | Reassigned development set; prior exposure and scorer limitations |
| Shared adapter: compatibility | Legacy accuracy **{shared['match_technical_compatibility']:.3f}** (8/8) | Scorer accepts opposite labels; prior exposure; not validated compatibility accuracy |
| Functional specialist | Accuracy **{m['FunctionalAccuracy']}** (23/28); macro-F1 **{m['FunctionalMacroF']}** | Exposed test; {m['ScenarioTestSeen']}/28 rows reuse training scenarios; 9 checkpoints inspected |
| Letter-label relevance specialist | Accuracy **{m['SingleAccuracy']}**; macro-F1 **{m['SingleMacroF']}** | 4,000 rows; balanced accuracy {m['SingleBalanced']}; complement recall {q['per_class']['complement']['recall']:.3f} |

The shared v1 rescore values **{m['SharedRelevance']} / {m['SharedIdentity']}** replace
the earlier headline pair 0.526 / 0.914. The earlier pair survives only in the
prior manuscript; it is not supported by a retained full evaluation vector.
Historical v2 identity also equals 0.914, but is a different run.

Extraction ties the strongest recorded API brand result. Its color F1 point
estimate is **{m['ExtractionColorDelta']} percentage points** above the strongest
recorded API color baseline (Haiku, F1 0.941176). This does **not** establish a
clean or statistically significant superiority claim: scoring is permissive,
two texts overlap training, and paired field predictions are missing.

Adapter and API relevance/identity scores use different samples. The functional
specialist and API results also use different evaluation sets. No cross-sample
frontier ranking is claimed. Historical API names are report labels, not verified
current endpoint snapshots. The retained eCeLLM functional run produced zero
strictly valid responses; its interface failure does not establish model inferiority.

The four adapters are distinct parameter states. Their combined task coverage
is not evidence that one shared model performs all tasks well. Human naturalness
validation, controlled forgetting/retention ablations, and universal retailer
suitability have not been demonstrated.

## Separate RunPod development baseline

This is **untuned Qwen3-1.7B**, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, not one of the four trained adapters.
It was evaluated on a new development suite with strict scoring, so its values
must not be subtracted from the historical adapter scores as an adaptation gain.

| Task / metric | Development result | Rows |
|---|---|---|
| Relevance accuracy / macro-F1 | {d['relevance']['accuracy']:.4f} / {d['relevance']['macro_f1']:.4f} | {d['relevance']['n']} |
| Identity accuracy / macro-F1 | {d['identity']['accuracy']:.4f} / {d['identity']['macro_f1']:.4f} | {d['identity']['n']} |
| Extraction exact brand / color F1 | {d['extraction']['exact']['brand']['f1']:.4f} / {d['extraction']['exact']['color']['f1']:.4f} | {d['extraction']['n']} |
| Extraction normalized brand / color F1 | {d['extraction']['normalized_exact']['brand']['f1']:.4f} / {d['extraction']['normalized_exact']['color']['f1']:.4f} | {d['extraction']['n']} |
| Extraction normalized joint accuracy | {d['extraction']['joint_accuracy']:.4f} | {d['extraction']['n']} |

All {manifest['rows']:,} rows have retained raw outputs. The evaluation loop took
{pilot['elapsed_seconds']:.2f} seconds on an RTX 3090, using batch size one,
bfloat16, greedy decoding, thinking disabled and a 64-token output cap.
This is sequential offline inference time, excluding model loading and provisioning;
it is **not** a concurrent serving benchmark or total campaign cost.
No new model training or frontier API calls were performed in this pilot.

## Serving, data quality and publication limits

The historical load-test CSVs contain **zero completed inference requests**.
Production throughput, tail latency and cost advantage remain unmeasured.
Neither historical token-cost assumptions nor the new offline inference timing
establish a production service-level claim.

New training remains blocked until independent human label/naturalness review
passes the quality gate. The new test suite has not been scored or used for
model selection. Machine checks and family proxies do not replace human review.

The paper is an empirical audit with a new development baseline and evaluation
infrastructure. It does not establish a new foundation-model architecture,
universal frontier superiority, or conference acceptance.

- [Detailed historical preprint]({GITHUB}/blob/main/paper/CommerceCore_Expansion_Paper.pdf)
- [Current anonymous SIGIR-style draft]({GITHUB}/blob/main/paper/CommerceCore_Expansion_SIGIR_Draft.pdf)
- [Alternative WSDM-style draft]({GITHUB}/blob/main/paper/CommerceCore_Expansion_WSDM_Draft.pdf)
- [ACM LaTeX source package]({GITHUB}/blob/main/paper/CommerceCore_Expansion_ACM_Source.zip)
- [Historical evidence audit]({GITHUB}/blob/main/paper/evidence/audit_results.json)
- [New development report and raw outputs]({GITHUB}/tree/main/{PILOT})
- [Publication gaps and related work]({GITHUB}/blob/main/paper/REVIEW_AND_PUBLICATION_STATUS.md)
"""
    common=common.replace('`',chr(96))
    template=(ROOT/'paper/README.template.md').read_text()
    if template.count('{{PUBLIC_RESULTS}}')!=1:
        raise ValueError('README template must contain exactly one public-results marker')
    generated={'README.md':template.replace('{{PUBLIC_RESULTS}}',common.rstrip())}
    titles={'understand-v1':'Brand/color extraction adapter',
            'match-v1':'Shared product-matching adapter',
            'functional-relation-v1':'Functional-relation specialist adapter',
            'relevance-singletoken-v1':'Letter-label relevance specialist adapter'}
    meta=read('paper/model_card_metadata.json')
    for slug in SLUGS:
        repo='arghya2030/commercecore-expansion-'+slug
        intro=f"---\n{meta[slug]}\n---\n\n# {titles[slug]}\n\nThis card describes **{repo}**. The weights are unchanged; this revision corrects\nand synchronizes the documented evidence. The tables identify which observations\nbelong to this adapter and which belong to other parameter states.\n\n"
        links=f"\n## Use and scope\n\nUse this adapter with its historical task-specific prompt interface; see the\n[GPU inference guide]({GITHUB}/blob/main/docs/INFERENCE.md). The new development\npilot uses an untuned base and a separate chat interface. The original\n[CommerceCore release](https://huggingface.co/arghya2030/commercecore-qwen3-1.7b)\nis a separate project and is not modified by this work.\n"
        generated['paper/model_cards/commercecore-expansion-'+slug+'.md']=intro+common+links
    dm=d['extraction']
    table="".join([
        f"Relevance & {d['relevance']['n']} & Accuracy & {d['relevance']['accuracy']:.4f} \\\\\n",
        f" & & Macro-F1 & {d['relevance']['macro_f1']:.4f} \\\\\n",
        f"Identity & {d['identity']['n']} & Accuracy & {d['identity']['accuracy']:.4f} \\\\\n",
        f" & & Macro-F1 & {d['identity']['macro_f1']:.4f} \\\\\n",
        f"Extraction & {dm['n']} & Brand F1 (exact) & {dm['exact']['brand']['f1']:.4f} \\\\\n",
        f" & & Color F1 (exact) & {dm['exact']['color']['f1']:.4f} \\\\\n",
        f" & & Brand F1 (norm.) & {dm['normalized_exact']['brand']['f1']:.4f} \\\\\n",
        f" & & Color F1 (norm.) & {dm['normalized_exact']['color']['f1']:.4f} \\\\\n",
        f" & & Joint acc. (norm.) & {dm['joint_accuracy']:.4f} \\\\\n"])
    generated['paper/acm/dev_rows.tex']=table
    summary={'audit_sha256':sha('paper/evidence/audit_results.json'),'historical_rounded_macros':m,
             'pilot_report_sha256':sha(PILOT+'/report.json'),'pilot_manifest_sha256':sha(PILOT+'/manifest.json'),
             'pilot_metrics':d,'scope':'Historical adapter scores and new untuned-base development scores are different experiments.'}
    generated['paper/evidence/public_results.json']=json.dumps(summary,indent=2)+'\n'
    return generated
def main():
    if platform.system()!='Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('Run publication generation/checks on RunPod only.')
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.is_file() or path.read_text()!=content:raise SystemExit('Public result drift: '+name)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print('Public README, model-card drafts and ACM result table agree with the saved evidence.')
if __name__=='__main__':main()
