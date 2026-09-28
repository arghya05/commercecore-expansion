# Historical inference interfaces

Execute on RunPod only. These examples document interfaces, not a new benchmark.
Pin each adapter revision before reproducible evaluation. Historical base loaders
did not consistently pin their downloads; the base pin below is prospective.


Each model uses its own adapter + prompt format. All four load the same base (`Qwen/Qwen3-1.7B`) via PEFT; only the repo ID, prompt, and label set change.

### Understand — brand/color extraction

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Run this on the RunPod GPU; no CPU fallback")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    torch_dtype=torch.bfloat16,
).to("cuda")
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-understand-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-understand-v1")

prompt = (
    'Extract the brand and color from this product listing. '
    'Respond with only a JSON object like {"brand": "...", "color": "..."}.\n'
    "Listing: Nike Air Max 270 Women's Trainers - Black/White. Breathable mesh upper.\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# {"brand": "Nike", "color": "Black/White"}
```

### Match — relevance

```python
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-match-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-match-v1")

prompt = (
    "Classify the query-product relevance: exact, substitute, complement, or irrelevant.\n"
    "Query: wireless bluetooth headphones\n"
    "Product: Sony WH-1000XM5 Wireless Noise Canceling Headphones\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# exact
```

### Match — identity

```python
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-match-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-match-v1")

prompt = (
    "Are these listings the same purchasable item or distinct?\n"
    "Listing A: Apple iPhone 15 128GB Blue\n"
    "Listing B: iPhone 15, 128GB, Blue - Unlocked\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# same
```

### Match — functional_relation

```python
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-functional-relation-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-functional-relation-v1")

prompt = (
    "Classify the functional relation: substitute, complement, or unrelated.\n"
    "Espresso Machine - Brew rich, full-bodied espresso shots for your favorite coffee drinks at home.\n"
    "Milk Frother - Create velvety steamed milk and foam for lattes, cappuccinos, and macchiatos.\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# complement
```
Example outputs illustrate the interface; they are not guaranteed.

### Match — technical_compatibility

```python
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-match-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-match-v1")

prompt = (
    "Classify technical compatibility: compatible, incompatible, or unknown.\n"
    "PS5 DualSense controller\nPS5 (2020 model)\nAnswer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# compatible
```
(Verified directly against the local adapter checkpoint. Not every example is correct — this same adapter also predicted `compatible` for a real `iPhone 14 case` / `iPhone 15` pair whose true label is `incompatible`, a genuine error found while preparing this example, not hidden from it. No serving route exists yet for this subtask — it's currently only reachable via direct model loading, not the REST API below.)

Historical REST API interface (execute on the RunPod machine):

```bash
pip install -r requirements.txt   # fastapi, uvicorn, torch, peft, transformers, bitsandbytes
uvicorn expansion.serve.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/v1/catalog/normalize \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "demo", "text": "Nike Air Max 270 Trainers - Black/White."}'
```

Production GPU throughput and cost have not been established. Do not execute inference on the laptop.

