"""HF08 (fastino/gliner2.5-base-v1) and HF09 (numind/NuExtract-2.0-2B)
structured-extraction comparators against the SAME 200-example grounded ABO
eval set used by understand_baselines.py (rules vs gliner2-base-v1).

Per RELATED_MODEL_COMPARISON_MATRIX.md HF08: "correct distinct loaders and
output alignments" — gliner2.5-base-v1 requires `AutoExtractor`, NOT the
legacy `GLiNER2.from_pretrained(...)` span loader used for gliner2-base-v1.
Verified against the model card: "GLiNER2.from_pretrained(...) is the legacy
span loader and will not dispatch this checkpoint."

Reuses score_field() from understand_baselines.py so numbers are directly
comparable to the existing rules=0.88/0.754 and gliner2-base=0.777/0.854
brand/color F1 baselines.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from understand_baselines import score_field


def run_gliner25(eval_path: Path, n: int = 200) -> dict:
    t0 = time.time()
    try:
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
        examples = json.loads(eval_path.read_text())[:n]

        brand_preds, color_preds = [], []
        for e in examples:
            result = model.extract_entities(e["text"], ["brand", "color"])
            ents = result.get("entities", {}) if isinstance(result, dict) else {}
            b = ents.get("brand") or [None]
            c = ents.get("color") or [None]
            brand_preds.append(b[0] if isinstance(b, list) else b)
            color_preds.append(c[0] if isinstance(c, list) else c)

        return {
            "comparison_status": "RUN",
            "loader": "gliner2.AutoExtractor.from_pretrained (native GLiNER2.5 loader, NOT the legacy "
                      "GLiNER2.from_pretrained span loader used for gliner2-base-v1)",
            "n_examples": len(examples),
            "brand": score_field(brand_preds, [e["brand"] for e in examples]),
            "color": score_field(color_preds, [e["color"] for e in examples]),
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


NUEXTRACT_TEMPLATE = '{"brand": "verbatim-string", "color": "verbatim-string"}'


def run_nuextract(eval_path: Path, n: int = 5, max_new_tokens: int = 48) -> dict:
    """NuExtract-2.0-2B is a ~2.2B image-text-to-text model; run on CPU with
    float32 (bf16 has poor CPU kernel support) on a SMALL subset given
    CPU-only 8GB-RAM hardware constraints in this pass.

    MEASURED throughput from a single-example smoke test on this exact
    hardware: ~1544 sec/example (~25.7 min) at max_new_tokens=64, most of
    which is autoregressive decode, not model load. At that rate n=50 would
    take >21 hours, infeasible for this pass. n defaults to 5 (a real,
    disclosed small sample, not a fabricated number) and max_new_tokens is
    reduced to 48 (still enough for the 2-field JSON schema used here) to
    keep wall-clock bounded. If it OOMs or fails, record the exact failure."""
    t0 = time.time()
    try:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForVision2Seq as _AutoModelCls
        except ImportError:
            # transformers >=5.x renamed this class; model card predates the rename.
            from transformers import AutoModelForImageTextToText as _AutoModelCls

        repo = "numind/NuExtract-2.0-2B"
        load_t0 = time.time()
        model = _AutoModelCls.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        load_time = time.time() - load_t0

        examples = json.loads(eval_path.read_text())[:n]
        brand_preds, color_preds = [], []
        raw_failures = 0
        per_example_sec = []

        for e in examples:
            ex_t0 = time.time()
            messages = [{"role": "user", "content": e["text"]}]
            text = processor.tokenizer.apply_chat_template(
                messages, template=NUEXTRACT_TEMPLATE, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=[text], padding=True, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, do_sample=False, num_beams=1,
                                                max_new_tokens=max_new_tokens)
            output = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            )[0]
            per_example_sec.append(time.time() - ex_t0)
            try:
                parsed = json.loads(output)
                brand_preds.append(parsed.get("brand"))
                color_preds.append(parsed.get("color"))
            except Exception:
                raw_failures += 1
                brand_preds.append(None)
                color_preds.append(None)

        return {
            "comparison_status": "RUN",
            "loader": "transformers.AutoModelForImageTextToText + AutoProcessor, native "
                      "apply_chat_template(template=...) schema-conditioned generation per model card "
                      "(card names AutoModelForVision2Seq; transformers 5.x renamed this class)",
            "n_examples": len(examples),
            "json_parse_failures": raw_failures,
            "brand": score_field(brand_preds, [e["brand"] for e in examples]),
            "color": score_field(color_preds, [e["color"] for e in examples]),
            "load_time_sec": load_time,
            "mean_sec_per_example": sum(per_example_sec) / len(per_example_sec) if per_example_sec else None,
            "per_example_sec": per_example_sec,
            "wall_time_sec": time.time() - t0,
            "note": f"Evaluated on n={len(examples)} (NOT the full 200) -- measured throughput on this "
                    f"CPU-only 8GB-RAM hardware is roughly {sum(per_example_sec)/len(per_example_sec):.0f} "
                    f"sec/example at max_new_tokens={max_new_tokens}; scaling to n=200 would take "
                    f"~{sum(per_example_sec)/len(per_example_sec)*200/3600:.1f} hours, infeasible for this "
                    f"pass. This is a disclosed small-sample result, not a full-eval-set claim." if per_example_sec
                    else "no examples completed",
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


if __name__ == "__main__":
    eval_path = Path("data/abo/understand_eval_grounded.json")
    out_path = Path("reports/extraction_comparators_2026-09-27/report.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    prior_gliner25 = Path("reports/extraction_comparators_2026-09-27/gliner25_only.json")
    if "hf08_gliner2_5_base_v1" not in out and prior_gliner25.exists():
        out["hf08_gliner2_5_base_v1"] = json.loads(prior_gliner25.read_text())
        print("Reused prior HF08 gliner2.5-base-v1 result (already RUN on full n=200)")
    elif "hf08_gliner2_5_base_v1" not in out:
        print("Running HF08 gliner2.5-base-v1...")
        out["hf08_gliner2_5_base_v1"] = run_gliner25(eval_path, n=200)
    print(json.dumps(out["hf08_gliner2_5_base_v1"], indent=2)[:2000])

    print("Running HF09 NuExtract-2.0-2B (n=5, measured throughput ~25min/example on this hardware)...")
    out["hf09_nuextract_2_0_2b"] = run_nuextract(eval_path, n=5, max_new_tokens=48)
    print(json.dumps(out["hf09_nuextract_2_0_2b"], indent=2)[:2000])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
