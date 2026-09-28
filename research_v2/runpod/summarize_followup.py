"""Generate paper and public documentation from verified follow-up artifacts."""
import json
from pathlib import Path
from research_v2.core import file_sha,paired_correctness
from research_v2.experiment import read_rows
from research_v2.execution import require_runpod

def main():
    require_runpod()
    root=Path(__file__).resolve().parents[2];out=root/'research_v2/results/followup_20260928'
    analysis=json.loads((out/'analysis.json').read_text())
    serving=json.loads((out/'serving/report.json').read_text())
    nue=json.loads((out/'nuextract/report.json').read_text())
    for name in ['base_legacy','base_chat','shared_legacy','shared_chat','understand_legacy','understand_chat','nuextract']:
        report=json.loads((out/name/'report.json').read_text())
        if report['manifest_sha256']!=file_sha(out/name/'manifest.json') or report['predictions_sha256']!=file_sha(out/name/'predictions.jsonl'):
            raise ValueError('Changed evidence '+name)
    if serving['request_sha256']!=file_sha(out/'serving/requests.jsonl'):raise ValueError('Changed serving evidence')
    md=['## Follow-up RunPod experiments — 2026-09-28','',
        'New **development-only** measurements; no retraining, final-test scoring or new frontier API calls. Base and adapters receive identical examples within each interface. All full paired runs use batch size eight, BF16, greedy decoding and a 64-token output cap.',
        '', '| Task | Interface | Base primary score | Adapter primary score | Base / adapter invalid rate |',
        '|---|---|---:|---:|---:|']
    tex=[]
    for task in ['relevance','identity','extraction']:
        for interface in ['legacy','chat']:
            r=analysis['results'][task+'_'+interface];metric='joint_accuracy' if task=='extraction' else 'macro_f1'
            invalid='parse_error_rate' if task=='extraction' else 'invalid_rate'
            b=r['base'][metric];a=r['adapter'][metric]
            md.append(f"| {task} | {interface} | {b:.4f} | {a:.4f} | {r['base'][invalid]:.4f} / {r['adapter'][invalid]:.4f} |")
            tex.append(f"{task.title()} & {interface} & {b:.4f} & {a:.4f} \\\\")
    n=nue['metrics']['extraction']
    extraction_rows=[r for r in read_rows(root/'research_v2/work/suite_v2/dev.jsonl') if r['task']=='extraction']
    nue_pairs={interface:paired_correctness(extraction_rows,
        read_rows(out/('understand_'+interface)/'predictions.jsonl'),read_rows(out/'nuextract/predictions.jsonl'))
        for interface in ['legacy','chat']}
    md.extend(['','Primary scores are macro-F1 for relevance/identity and normalized exact joint accuracy for extraction. Legacy uses the original training completion prompts; chat uses the common strict task prompt. Interface differences are part of the experiment, not hidden preprocessing.',
        '',f"NuExtract-2.0-2B, using its native text/schema interface on the same 400 extraction rows: normalized joint accuracy **{n['joint_accuracy']:.4f}**, normalized brand/color F1 **{n['normalized_exact']['brand']['f1']:.4f} / {n['normalized_exact']['color']['f1']:.4f}**, invalid rate **{n['parse_error_rate']:.4f}**. Its immutable revision, schema and raw outputs are recorded. This is one comparator/interface, not a comprehensive leaderboard.",
        '', 'The known-source audit found zero exact-text matches for these development rows; identity entity-level overlap remains unknown. These retrospective adapter diagnostics do not establish an independent final-test win. Raw outputs, per-class scores and exploratory paired correctness intervals are retained.',
        '', '### HTTP serving prototype', '', '| Batch limit | Concurrency | Completed req/s | Correct fraction | Mean run p95 (s) | GPU-rate estimate / 1,000 valid requests |',
        '|---:|---:|---:|---:|---:|---:|'])
    serving_tex=[];aggregates=[]
    for limit in [1,8]:
        for concurrency in [1,4,8]:
            cells=[r for r in serving['cells'] if r['batch_limit']==limit and r['concurrency']==concurrency]
            attempted=sum(r['attempted'] for r in cells);completed=sum(r['completed'] for r in cells)
            valid=sum(r['valid'] for r in cells);correct=sum(r['correct'] for r in cells);elapsed=sum(r['elapsed_seconds'] for r in cells)
            cost=sum(r['gpu_quote_cost_usd'] for r in cells);qps=completed/elapsed;p95=sum(r['p95_seconds'] for r in cells)/len(cells)
            costvalid=1000*cost/valid if valid else None
            md.append(f"| {limit} | {concurrency} | {qps:.2f} | {correct/attempted:.4f} | {p95:.3f} | "+(f"${costvalid:.5f}" if costvalid is not None else 'undefined')+' |')
            serving_tex.append(f"{limit} & {concurrency} & {qps:.2f} & {correct/attempted:.3f} & {p95:.3f} \\\\")
            aggregates.append({'batch_limit':limit,'concurrency':concurrency,'attempted':attempted,'completed':completed,'valid':valid,'correct':correct,'elapsed_seconds':elapsed,'qps':qps,'mean_run_p95_seconds':p95,'gpu_rate_cost_per_1000_valid_usd':costvalid})
    md.extend(['','One RTX 3090; same-pod loopback HTTP client/server; shared identity adapter; three repeats of 200 requests per cell; 50 warmup requests excluded per engine/repeat. Batching waits at most 5 ms, with an eight-token output cap. All responses and errors are retained. These are closed-loop, single-task prototype measurements, not an optimized production server or open-loop SLO test. Tail estimates are exploratory.',
        '', 'Cost uses the $0.22/hour GPU quote during measured windows. It excludes setup, idle time, storage and network; it is not a full invoice or frontier-cost advantage. No new architecture, controlled training improvement, universal frontier win or conference acceptance is established.',
        '', 'A slow batch-one run was interrupted for runtime reasons before completing; its journal is preserved and excluded from full-run tables. The planned large serving campaign was narrowed to 200 requests/cell/repeat; broader hardware, public-benchmark, human-review and controlled-training experiments remain open.',
        '', '[Raw follow-up evidence](https://github.com/arghya05/commercecore-expansion/tree/main/research_v2/results/followup_20260928) · [Experiment protocol](https://github.com/arghya05/commercecore-expansion/blob/main/research_v2/EXPERIMENT_CAMPAIGN.md)', ''])
    section=r'''\section{Matched Development and HTTP Follow-up}
The follow-up evaluates the pinned base and existing adapters on the same
1,495 development rows under two interfaces: original training completion
prompts (legacy), and the common chat task specification. All complete paired
runs use bfloat16, batch size eight, greedy decoding, a 64-token output cap and
the strict scorer. Neither training nor final-test scoring occurs. An initial
batch-one run was interrupted for runtime reasons; its partial journal is
retained but excluded. Batching changes are shared by all paired candidates.

\begin{table}[t]
\centering\small
\caption{Matched development primary scores: macro-F1 for relevance and identity;
normalized exact joint accuracy for extraction. Identical rows within each task.}
\label{tab:followup}
\begin{tabular}{llrr}\toprule
Task & Interface & Base & Adapter\\\midrule
'''+ '\n'.join(tex)+r'''
\bottomrule\end{tabular}\end{table}

Interface and adaptation must be interpreted jointly (Table~\ref{tab:followup}).
Strict schema failures remain in the denominator; per-class and validity
results accompany the raw outputs. The recorded historical training-source
audit finds no exact-text matches on these development rows, but identity
entity exposure remains unresolved. Thus this is a retrospective development
comparison, not certified independent generalization. Paired binary/joint
correctness diagnostics use connected groups and 2,000 exploratory bootstrap
resamples; their intervals do not describe macro-F1 effects. No repeated-seed
training effect or matched frontier-API ranking is established.

NuExtract-2.0-2B is evaluated using its native text/schema interface, with
verbatim-string brand/color fields and no demonstrations or prompt search.
On the same 400 extraction rows it obtains normalized joint accuracy
'''+f"{n['joint_accuracy']:.4f}"+r''', normalized brand/color F1
'''+f"{n['normalized_exact']['brand']['f1']:.4f}/{n['normalized_exact']['color']['f1']:.4f}"+r''', and invalid rate
'''+f"{n['parse_error_rate']:.4f}"+r'''. The revision, prompt and raw outputs are
retained. This closes one missing comparator run, not broader baseline coverage.

\begin{table}[t]
\centering\small
\caption{GPU loopback HTTP prototype. B: maximum batch size; C: client
concurrency. QPS counts completed requests; correctness is over all attempts.
p95 is the mean of three within-run percentiles, not a pooled percentile.}
\label{tab:httpfollowup}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rrrrr}\toprule
B & C & QPS & Correct & p95 (s)\\\midrule
'''+ '\n'.join(serving_tex)+r'''
\bottomrule\end{tabular}\end{table}

The shared identity adapter is served through a bounded HTTP queue on one
RTX 3090, comparing batch limits one and eight with at most 5\,ms batching wait.
Each cell has three repetitions of 200 requests on the same development trace;
50 warmup requests per engine/repetition are excluded. Generation permits eight
tokens. Client and server share the pod, and the load is closed-loop, so these
results do not identify open-loop overload behavior or external network latency.
The smaller workload is an explicit budgeted departure from the prospective
larger serving study; tail estimates are exploratory.

Quality and throughput are reported together (Table~\ref{tab:httpfollowup});
all attempted responses, errors and timestamps are retained. Window-specific
cost estimates use the quoted GPU rate of \$0.22/hour and exclude provisioning,
idle time, storage and network. They are not invoices, production SLO costs or
an API cost advantage. The reference queue is not a new serving algorithm or
a comparison with an established optimized inference engine.
'''
    section=section.replace('NuExtract-2.0-2B is evaluated using its native text/schema interface, with',
        'The NuExtract-2.0-2B comparison uses the native schema interface for text, with')
    rel=analysis['results']['relevance_chat'];ident=analysis['results']['identity_chat'];ext=analysis['results']['extraction_chat']
    if not all(analysis['results'][t+'_legacy']['base'].get('invalid_rate',analysis['results'][t+'_legacy']['base'].get('parse_error_rate'))==1 for t in ['relevance','identity','extraction']):
        raise ValueError('Legacy interpretation must be revised for changed evidence')
    if any(analysis['results']['relevance_'+i]['adapter']['per_class']['complement']['predicted']!=0 for i in ['legacy','chat']):
        raise ValueError('Minority-label interpretation must be revised')
    interpretation=rf'''The unadapted base fails the strict output contract on every legacy-prompt
row. Its zero scores measure interface failure, not absence of commerce knowledge;
the chat comparison is essential to interpretation. Under chat, relevance
macro-F1 changes only from {rel['base']['macro_f1']:.4f} to {rel['adapter']['macro_f1']:.4f}, whereas identity changes from {ident['base']['macro_f1']:.4f}
to {ident['adapter']['macro_f1']:.4f}. The adapter predicts no complement labels in either interface.
Extraction joint accuracy changes from {ext['base']['joint_accuracy']:.4f} to {ext['adapter']['joint_accuracy']:.4f} under chat.
The connected grouping yields {rel['paired_adapter_minus_base']['groups']} relevance, {ident['paired_adapter_minus_base']['groups']} identity and {ext['paired_adapter_minus_base']['groups']} extraction
components. We suppress the identity bootstrap interval because fewer than 20 components
do not support that uncertainty estimate. BF16 greedy outputs can depend on
batching; the batch-eight base is rerun rather than substituted by the earlier
batch-one pilot. These findings distinguish task adaptation, label compliance
and interface sensitivity rather than supporting a universal gain.

'''
    section=section.replace('The NuExtract-2.0-2B comparison',interpretation+'The NuExtract-2.0-2B comparison')
    md.extend(['The base produced invalid outputs for every legacy-prompt row; those zero scores diagnose interface failure, not absent task knowledge. The chat comparison is therefore essential. Identity has only five connected evaluation components, so its cluster-bootstrap interval is suppressed. The shared adapter predicts no complement labels in either interface. Batch-eight base scores are separate from the earlier batch-one pilot.',''])
    (root/'paper/evidence/followup.tex').write_text(section+'\n')
    (root/'paper/evidence/followup_summary.md').write_text('\n'.join(md))
    (root/'paper/evidence/followup_public_results.json').write_text(json.dumps({'analysis_sha256':file_sha(out/'analysis.json'),
        'serving_sha256':file_sha(out/'serving/report.json'),'nuextract_sha256':file_sha(out/'nuextract/report.json'),
        'matched':analysis['results'],'nuextract':n,'adapter_minus_nuextract_paired_correctness':nue_pairs,'serving':aggregates},indent=2)+'\n')

if __name__=='__main__':main()
