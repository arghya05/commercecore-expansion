# RunPod-only execution

All tests, dependency installs, data preparation, inference, training, serving
measurements, rendering and PDF compilation run on RunPod. The laptop only
connects, transfers files and publishes artifacts. The environment guards reject
accidental laptop execution; they are not an access-control system.

## Pilot performed on 2026-09-28

The user approved a **$25 maximum initial pilot**. The dedicated on-demand pod
used one RTX 3090 (24 GiB), quoted at $0.22/hour, with a provider stop deadline
two hours after creation. Storage charges are separate. A quoted hourly rate
is not an invoice, and a job timeout alone does not stop pod billing.

The pilot used an isolated `/workspace/commercecore_expansion_research_v2`
directory and a dedicated Python environment. Model weights were downloaded
directly to the pod. No training or paid frontier API calls were made.
The 9-row GPU smoke run and the full 1,495-row development baseline completed.
New protocol tests (16) and historical evidence tests (7) passed remotely.
Both ACM drafts and their isolated source package compiled on RunPod.

See [saved outputs and environment](../results/runpod_pilot_20260928/).
The development loop's 218.90 seconds excludes loading and provisioning and
does not measure production serving performance. Data quality review remains
a gate before any future training. Synthetic tasks are not yet approved.

## Reuse safely

1. Check actual on-demand capacity, CUDA compatibility and total GPU/storage
   pricing. The selected pilot instance is not proven optimal.
2. Use a dedicated pod and directory, explicit total budget and provider-side
   shutdown deadline. Record state outside the pod.
3. Transfer only required inputs at a limited rate; download weights remotely.
4. Set the actual `RUNPOD_POD_ID` in the remote SSH shell, install the pinned
   environment with `setup.sh`, and run tests there.
5. Use `dev_pilot.py` with fresh job IDs. Its default checks nine development
   rows; `CC_FULL_DEV=1` runs the complete development contract.
6. Copy logs, predictions, hashes, PDFs and sources back, then verify delivery.
7. Terminate only the dedicated pilot after backup. A stopped pod can continue
   incurring storage charges; terminating it deletes its pod volume.

The API helper `control.py` performs only cloud lifecycle control and uses a
saved local RunPod credential. It never loads a model or launches local tests.
The historical full-suite `pilot.sh` is retained; `dev_pilot.py` was used for
this pilot and avoids waiting on training/test transfer for development inference.

Hugging Face card access and publishing run on the pod. Credentials arrive over
SSH stdin, are not logged or saved, and are used only for the four Expansion
cards. Publication verifies that weights and repository visibility are unchanged.
The original CommerceCore model is outside this scope.

[RunPod connection documentation](https://docs.runpod.io/pods/connect-to-a-pod)
and [pod management](https://docs.runpod.io/pods/manage-pods) explain connection
and lifecycle behavior. On-demand availability does not guarantee no interruption.
