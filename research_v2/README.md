# Isolated research protocol, version 2

All research execution is **RunPod only**: data preparation, tests, model runs,
evaluation, serving measurements, and PDF compilation. The laptop is a connection,
file-transfer and GitHub publishing client. Do not install dependencies or run
research commands on the laptop.

This directory is separate from the historical `expansion/` runtime and from
the original CommerceCore repository/model release. Historical source, data and
reports are preserved. The new code does not import the original CommerceCore
model or overwrite historical training artifacts.

## Evidence and gates

The real-source suite has 17,997 training, 1,495 development and 3,000 test rows
across relevance, identity and brand/color extraction. It uses ESCI, WDC Products
and ABO subsets; these are **not official benchmark/leaderboard scores**.
See [the exact manifest](manifests/real_sources_v2.json).

Machine checks cover contracts, labels, hashes and cross-partition group overlap.
Family grouping uses metadata proxies, not verified semantic family annotation.
Extraction covers literal, nonempty brand/color values. Missing attributes,
languages and retailer-specific distributions need separate coverage.

**Training is blocked** until independent human label/naturalness review passes
`experiment.training_gate`. No approval file or human measurements have been
invented. Functional-relation and technical-compatibility data remain blocked
pending new, independently reviewed scenario-disjoint data.

The pinned Qwen3-1.7B development baseline is a protocol pilot, not a new
fine-tuned model or proof of improvement. Test data is not used for model
selection. The verifier reads test labels only for structural integrity, which
is distinct from scoring model outputs on them.

## Execute on the provisioned RunPod

Use the recorded environment and set the actual pod ID in the remote shell.
SSH shells may not inherit it. The check prevents accidental laptop execution,
not a malicious caller.

```sh
export RUNPOD_POD_ID=<actual-pod-id>
/workspace/cc-research-venv/bin/python -m unittest discover -s research_v2/tests -v
/workspace/cc-research-venv/bin/python -m research_v2.experiment verify
CC_JOB_ID=qwen_base_gpu_smoke_v2 /workspace/cc-research-venv/bin/python -m research_v2.runpod.dev_pilot
CC_JOB_ID=qwen_base_gpu_dev_v2 CC_FULL_DEV=1 /workspace/cc-research-venv/bin/python -m research_v2.runpod.dev_pilot
```

Use a fresh job ID for each run. `run_local.py` retains its earlier filename
but refuses non-RunPod execution and has no CPU fallback. Oversized inputs are
rejected, never silently truncated; failed rows remain in metric denominators.
A nine-row smoke test has no quality-estimation interpretation.

Freeze the complete candidate cohort, model identities, prompts, decoding and
development selection reports before final testing. `freeze` records this
contract; `open_test` permits one recorded evaluation access. These are
tamper-evident workflow controls, not an access-control service. Test outcomes
must not trigger retraining within the same reported experiment.

Read [RunPod operations](runpod/README.md) and the
[publication review](../paper/REVIEW_AND_PUBLICATION_STATUS.md) before extending
the campaign. Sequential inference timings are not production serving throughput.
