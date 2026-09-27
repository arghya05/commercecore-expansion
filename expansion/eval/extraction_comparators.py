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


def run_nuextract(eval_path: Path, n: int = 50) -> dict:
    """NuExtract-2.0-2B is a ~2.2B image-text-to-text model; run on CPU with
    float32 (bf16 has poor CPU kernel support) on a SMALL subset (n=50) given
    CPU-only 8GB-RAM hardware constraints in this pass. If it OOMs or the
    vision-conditioned processor requires an image input we cannot supply,
    record the exact failure."""
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
        model = _AutoModelCls.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

        examples = json.loads(eval_path.read_text())[:n]
        brand_preds, color_preds = [], []
        raw_failures = 0

        for e in examples:
            messages = [{"role": "user", "content": e["text"]}]
            text = processor.tokenizer.apply_chat_template(
                messages, template=NUEXTRACT_TEMPLATE, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=[text], padding=True, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=128)
            output = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            )[0]
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
            "loader": "transformers.AutoModelForVision2Seq + AutoProcessor, native chat_template(template=...) "
                      "schema-conditioned generation per model card",
            "n_examples": len(examples),
            "json_parse_failures": raw_failures,
            "brand": score_field(brand_preds, [e["brand"] for e in examples]),
            "color": score_field(color_preds, [e["color"] for e in examples]),
            "wall_time_sec": time.time() - t0,
            "note": "Evaluated on n=50 (not the full 200) due to CPU-only float32 generation cost on "
                    "8GB-RAM hardware in this pass; see benchmark_cells.json for full disclosure.",
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


if __name__ == "__main__":
    eval_path = Path("data/abo/understand_eval_grounded.json")
    out = {}

    print("Running HF08 gliner2.5-base-v1...")
    out["hf08_gliner2_5_base_v1"] = run_gliner25(eval_path, n=200)
    print(json.dumps(out["hf08_gliner2_5_base_v1"], indent=2)[:2000])

    print("Running HF09 NuExtract-2.0-2B (n=50)...")
    out["hf09_nuextract_2_0_2b"] = run_nuextract(eval_path, n=50)
    print(json.dumps(out["hf09_nuextract_2_0_2b"], indent=2)[:2000])

    out_path = Path("reports/extraction_comparators_2026-09-27/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
