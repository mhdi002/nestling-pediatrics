from pathlib import Path

path = Path("assistant/agent/orchestrator.py")
text = path.read_text(encoding="utf-8")
old = """        content = build_xlam_prompt(focused)
        messages = [{\"role\": \"user\", \"content\": content}]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors=\"pt\"
        )
        device = next(self._model.parameters()).device
        if hasattr(inputs, \"to\"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model.generate(
            inputs if not isinstance(inputs, dict) else inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self._tok.eos_token_id,
        )
        in_len = inputs.shape[1] if hasattr(inputs, \"shape\") else inputs[\"input_ids\"].shape[1]
        text = self._tok.decode(outputs[0][in_len:], skip_special_tokens=True)
"""
new = """        content = build_xlam_prompt(focused)
        messages = [{\"role\": \"user\", \"content\": content}]
        # transformers>=5 may return BatchEncoding; tokenize then **kwargs to generate
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self._model.parameters()).device
        model_inputs = self._tok(prompt, return_tensors=\"pt\")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        outputs = self._model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self._tok.eos_token_id,
        )
        in_len = model_inputs[\"input_ids\"].shape[1]
        text = self._tok.decode(outputs[0][in_len:], skip_special_tokens=True)
"""
if old not in text:
    print("OLD NOT FOUND")
    idx = text.find("build_xlam_prompt(focused)")
    print(repr(text[idx:idx+1100]))
    raise SystemExit(1)
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched OK")
