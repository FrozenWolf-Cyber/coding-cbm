import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering, top_k_top_p_filtering_batched


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck-mode plumbing (used when CBL/CBLResidual.cbl_layer_idx >= 0).
# Insert the CBL between Llama decoder layer L and L+1; the rest of the model +
# original lm_head produce the vocab logits naturally.
# ─────────────────────────────────────────────────────────────────────────────


def _get_llama_model(preLM):
    """Return the underlying ``LlamaModel`` regardless of PEFT wrapping.

    Handles three layouts seen in this repo:
      - raw ``LlamaModel`` (eval after ``load_adapter`` on a fresh LlamaModel)
      - PEFT-wrapped ``LlamaModel`` (training: ``preLM = get_peft_model(LlamaModel, ...)``)
      - ``LlamaForCausalLM`` (just in case): the inner ``LlamaModel`` is at ``.model``.
    """
    base = preLM
    # Drill through PEFT wrapping (PeftModel.base_model.model -> original model).
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    # If we landed on a CausalLM head wrapper, descend once more to the LlamaModel.
    if not (hasattr(base, "layers") and hasattr(base, "norm")) and hasattr(base, "model"):
        base = base.model
    if not (hasattr(base, "layers") and hasattr(base, "norm")):
        raise RuntimeError(
            "Could not locate Llama decoder layers + final norm on the given preLM "
            f"(type={type(preLM).__name__})."
        )
    return base


def _bottleneck_forward_tail(llama_model, h_L_out, attention_mask, layer_idx):
    """Run ``layers[layer_idx+1:] + norm`` on ``h_L_out``. Returns ``h_final``.

    Mirrors HuggingFace ``LlamaModel.forward`` for the tail of the stack so we
    can re-run the post-bottleneck portion on a modified hidden state (used for
    --intervention_gen_loss in intermediate mode).
    """
    bsz, seq_len, _ = h_L_out.shape
    device = h_L_out.device
    cache_position = torch.arange(seq_len, device=device)
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    else:
        position_ids = cache_position.unsqueeze(0).expand(bsz, -1)
    # HF >= ~4.46: `_update_causal_mask` removed from `LlamaModel`; use `create_causal_mask` instead.
    if hasattr(llama_model, "_update_causal_mask"):
        causal_mask = llama_model._update_causal_mask(
            attention_mask, h_L_out, cache_position, None, False,
        )
    else:
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=llama_model.config,
            input_embeds=h_L_out,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
    position_embeddings = llama_model.rotary_emb(h_L_out, position_ids)
    h = h_L_out
    for layer in llama_model.layers[layer_idx + 1:]:
        h = layer(
            h,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    return llama_model.norm(h)


def _bottleneck_step_logits(
    cbl,
    preLM,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    intervene_concepts_fn=None,
    llama_vocab_weight=None,
):
    """One forward step that handles both last-layer and intermediate modes.

    Returns ``(logits, concepts_relu, past_key_values, last_hidden_state)``.

    - ``cbl.cbl_layer_idx == -1``: classic path — preLM forward, ``cbl/unsup/fc``
      on the last hidden state, optional ``+ llama_logits``.
    - ``cbl.cbl_layer_idx >= 0``: register a forward hook on layers[L] that runs
      the bottleneck (with optional intervention) and substitutes its output;
      vocab logits come from ``F.linear(last_hidden_state, llama_vocab_weight)``.

    ``intervene_concepts_fn`` (optional): callable mutating the *raw* (pre-ReLU)
    concepts in-place or returning a new tensor; matches the existing
    in-place mutation semantics in the per-variant generate functions.
    """
    use_cache = past_key_values is not None or True
    fwd_kwargs = {"use_cache": True}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask
    if past_key_values is not None:
        fwd_kwargs["past_key_values"] = past_key_values

    if cbl.cbl_layer_idx == -1:
        outputs = preLM(input_ids, **fwd_kwargs)
        last_hidden = outputs.last_hidden_state
        features = last_hidden.float()
        concepts = cbl.cbl(features)
        unsup_features = cbl._unsup_branch(features)
        if intervene_concepts_fn is not None:
            concepts = intervene_concepts_fn(concepts)
        logits = cbl.fc(torch.cat((cbl.relu(concepts), unsup_features), dim=-1))
        if llama_vocab_weight is not None:
            llama_logits = F.linear(last_hidden.to(llama_vocab_weight.dtype), llama_vocab_weight)
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits, cbl.relu(concepts), getattr(outputs, "past_key_values", past_key_values), last_hidden

    # Intermediate (bottleneck) mode.
    if llama_vocab_weight is None:
        raise ValueError("llama_vocab_weight is required when cbl_layer_idx >= 0 (no fallback lm_head).")
    llama_model = _get_llama_model(preLM)
    target_layer = llama_model.layers[cbl.cbl_layer_idx]
    store: dict = {}

    def _hook(module, args, kwargs, output):
        h_L = output[0] if isinstance(output, tuple) else output
        h_L_in_dtype = h_L.dtype
        feats = h_L.float()
        concepts = cbl.cbl(feats)
        unsup = cbl._unsup_branch(feats)
        if intervene_concepts_fn is not None:
            concepts = intervene_concepts_fn(concepts)
        concepts_relu = cbl.relu(concepts)
        h_L_proj = cbl.proj(torch.cat((concepts_relu, unsup), dim=-1)).to(h_L_in_dtype)
        if cbl.use_residual:
            h_L_out = h_L_proj + h_L
        else:
            h_L_out = h_L_proj
        store["concepts"] = concepts_relu
        store["unsup"] = unsup
        store["h_L"] = h_L
        store["h_L_proj"] = h_L_proj
        if isinstance(output, tuple):
            return (h_L_out,) + tuple(output[1:])
        return h_L_out

    handle = target_layer.register_forward_hook(_hook, with_kwargs=True)
    try:
        outputs = preLM(input_ids, **fwd_kwargs)
    finally:
        handle.remove()
    last_hidden = outputs.last_hidden_state
    logits = F.linear(last_hidden.to(llama_vocab_weight.dtype), llama_vocab_weight)
    return logits, store["concepts"], getattr(outputs, "past_key_values", past_key_values), last_hidden


def _safe_multinomial_from_logits(filtered_logits: torch.Tensor) -> torch.Tensor:
    """Sample from logits robustly.

    Prevents CUDA device-side asserts in ``torch.multinomial`` when the
    probability tensor contains NaN/Inf or sums to zero (e.g. all tokens were
    filtered to -inf).
    """
    # Fast path: identical to the original code when it is well-defined.
    probs_orig = torch.softmax(filtered_logits, dim=-1)
    denom_orig = probs_orig.sum(dim=-1)
    if torch.isfinite(probs_orig).all() and torch.isfinite(denom_orig).all() and (denom_orig > 0).all():
        return torch.multinomial(probs_orig, num_samples=1)

    # Fallback: sanitize probabilities to avoid device-side asserts.
    logits_f = filtered_logits.float()
    probs = torch.softmax(logits_f, dim=-1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    probs = torch.clamp(probs, min=0.0)

    denom = probs.sum(dim=-1, keepdim=True)
    safe_denom = torch.where(torch.isfinite(denom), denom, torch.zeros_like(denom))
    probs = probs / safe_denom.clamp_min(1e-20)

    bad_rows = (safe_denom <= 0).squeeze(-1)
    if bad_rows.any():
        safe_logits = torch.where(torch.isfinite(logits_f), logits_f, torch.full_like(logits_f, -1e9))
        argmax = torch.argmax(safe_logits, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(-1, argmax, 1.0)
        probs = torch.where(bad_rows.unsqueeze(-1), one_hot, probs)

    return torch.multinomial(probs, num_samples=1)

class Roberta_classifier(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(128, class_num)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Llama_baseline(nn.Module):
    def __init__(self, config, class_num):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, 128)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(128, class_num)

    def forward(self, t):
        projected = self.projection(t)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Llama_baseline_generation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, 768)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(768, config.vocab_size)

    def forward(self, t, llama_logits=None):
        projected = self.projection(t)
        x = self.gelu(projected)
        x = self.dropout(x)
        logits = self.fc(x)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits

    def generate(self, ids, preLM, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001, llama_vocab_weight=None):
        past_key_values = None
        for i in range(length):
            outputs = preLM(ids[:, -1:] if past_key_values is not None else ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            projected = self.projection(features)
            x = self.gelu(projected)
            x = self.dropout(x)
            logits = self.fc(x)
            if llama_vocab_weight is not None:
                llama_logits = F.linear(outputs.last_hidden_state.to(llama_vocab_weight.dtype), llama_vocab_weight)
                logits = logits + llama_logits.to(dtype=logits.dtype)
            score = logits[:, -1, ids[0]]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[:, -1, ids[0]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return ids

class CBL(nn.Module):
    def __init__(self, config, concept_dim, tokenizer, cbl_layer_idx: int = -1, use_residual: bool = True):
        super().__init__()
        self.cbl = nn.Linear(config.hidden_size, concept_dim)
        self.unsup = nn.Linear(config.hidden_size, 768)
        self.fc = nn.Linear(concept_dim + 768, config.vocab_size)
        self.relu = nn.ReLU()
        self.concept_dim = concept_dim
        self.tokenizer = tokenizer
        self.match_layer = None
        if concept_dim != 768:
            print("Warning: concept_dim and unsup feature dim are not equal so creating a linear layer to match dimensions.")
            self.match_layer = nn.Linear(768, concept_dim)
        # Bottleneck (intermediate-layer) wiring. Only allocated when requested
        # so last-layer-mode checkpoints stay byte-identical to before.
        self.cbl_layer_idx = int(cbl_layer_idx)
        self.use_residual = bool(use_residual)
        self.hidden_size = config.hidden_size
        self.proj = None
        if self.cbl_layer_idx >= 0:
            self.proj = nn.Linear(concept_dim + 768, config.hidden_size)

    @property
    def _unsup_branch(self):
        return self.unsup

    def forward(self, features, llama_logits=None):
        concepts = self.cbl(features)
        unsup_features = self.unsup(features)
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return self.relu(concepts), unsup_features, logits, self.match_layer(unsup_features) if self.match_layer else unsup_features

    def forward_full(self, preLM, input_ids, attention_mask=None, llama_vocab_weight=None):
        """Run the full backbone + CBL/bottleneck. Mode is selected by ``cbl_layer_idx``.

        Returns ``(concepts_relu, unsup, vocabs, matched_unsup, h_L, h_L_proj)``.
        ``h_L`` and ``h_L_proj`` are ``None`` in last-layer mode.
        """
        if self.cbl_layer_idx == -1:
            features = preLM(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            llama_logits = (
                F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            )
            concepts, unsup, vocabs, matched = self.forward(features.float(), llama_logits=llama_logits)
            return concepts, unsup, vocabs, matched, None, None
        if llama_vocab_weight is None:
            raise ValueError("llama_vocab_weight is required when cbl_layer_idx >= 0.")
        llama_model = _get_llama_model(preLM)
        target_layer = llama_model.layers[self.cbl_layer_idx]
        store: dict = {}

        def _hook(module, args, kwargs, output):
            h_L = output[0] if isinstance(output, tuple) else output
            feats = h_L.float()
            concepts_raw = self.cbl(feats)
            unsup = self.unsup(feats)
            concepts_relu = self.relu(concepts_raw)
            h_L_proj_f = self.proj(torch.cat((concepts_relu, unsup), dim=-1))
            h_L_proj = h_L_proj_f.to(h_L.dtype)
            if self.use_residual:
                h_L_out = h_L_proj + h_L
            else:
                h_L_out = h_L_proj
            store["concepts"] = concepts_relu
            store["unsup"] = unsup
            store["h_L"] = h_L
            store["h_L_proj"] = h_L_proj_f
            if isinstance(output, tuple):
                return (h_L_out,) + tuple(output[1:])
            return h_L_out

        handle = target_layer.register_forward_hook(_hook, with_kwargs=True)
        try:
            outputs = preLM(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        h_final = outputs.last_hidden_state
        vocabs = F.linear(h_final.to(llama_vocab_weight.dtype), llama_vocab_weight)
        unsup = store["unsup"]
        matched = self.match_layer(unsup) if self.match_layer else unsup
        return store["concepts"], unsup, vocabs, matched, store["h_L"], store["h_L_proj"]

    def intervene(self, unsup_features, intervene, llama_logits=None):
        concepts = intervene
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits

    def intervene_full(self, preLM, h_L, attention_mask, intervened_concepts, unsup_features, llama_vocab_weight):
        """Tail-only re-forward for intervention loss in intermediate mode.

        Reuses the cached ``h_L`` (from the un-intervened forward) so we do *not*
        rerun layers[0..L]; only ``proj``, layers[L+1:], the final norm and lm_head.
        """
        if self.cbl_layer_idx == -1:
            raise RuntimeError("intervene_full is only valid in intermediate mode (cbl_layer_idx >= 0).")
        h_L_proj_f = self.proj(torch.cat((self.relu(intervened_concepts), unsup_features), dim=-1))
        h_L_proj = h_L_proj_f.to(h_L.dtype)
        h_L_out = (h_L_proj + h_L) if self.use_residual else h_L_proj
        llama_model = _get_llama_model(preLM)
        h_final = _bottleneck_forward_tail(llama_model, h_L_out, attention_mask, self.cbl_layer_idx)
        return F.linear(h_final.to(llama_vocab_weight.dtype), llama_vocab_weight)

    def generate(self, ids, preLM, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001, llama_vocab_weight=None):
        past_key_values = None

        def _intervene_fn(concepts):
            if intervene:
                for j in range(self.concept_dim):
                    concepts[0, :, j] = intervene[j]
            return concepts

        concepts = None
        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                intervene_concepts_fn=_intervene_fn if intervene else None,
                llama_vocab_weight=llama_vocab_weight,
            )
            score = logits[:, -1, ids[0]]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[:, -1, ids[0]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return ids, concepts[0] if concepts is not None else None

    def generate_batch(
        self,
        ids,
        preLM,
        num_samples=1,
        intervene=None,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """Generate num_samples trajectories in parallel (batched autoregressive)."""
        ids = ids.expand(num_samples, -1).contiguous()  # (B, prompt_len)
        finished = torch.zeros(num_samples, dtype=torch.bool, device=ids.device)
        past_key_values = None
        concepts = None

        def _intervene_fn(concepts):
            if intervene and not keep_other_concepts:
                for j in range(self.concept_dim):
                    concepts[:, :, j] = intervene[j]
            elif intervene and keep_other_concepts:
                for j in range(self.concept_dim):
                    val = intervene[j]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if val != 0:
                        concepts[:, :, j] = val
            return concepts

        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                intervene_concepts_fn=_intervene_fn if intervene else None,
                llama_vocab_weight=llama_vocab_weight,
            )
            for b in range(num_samples):
                if not finished[b]:
                    score = logits[b, -1, ids[b]].clone()
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, -1, ids[b]] = score
            next_token_logits = logits[:, -1, :] / temp  # (B, vocab_size)
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)  # (B, 1)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return ids, concepts if concepts is not None else None

    def generate_intervention_batch_parallel(
        self,
        ids,
        preLM,
        attention_mask,
        num_samples=1,
        interventions=None,
        intervention_mask=None,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """Generate for batched prompts in parallel with optional per-row interventions."""
        prompt_batch = ids.size(0)
        if prompt_batch > 1 and num_samples > 1:
            ids = ids.repeat_interleave(num_samples, dim=0).contiguous()
            attention_mask = attention_mask.repeat_interleave(num_samples, dim=0).contiguous()
        elif prompt_batch == 1 and num_samples > 1:
            ids = ids.expand(num_samples, -1).contiguous()
            attention_mask = attention_mask.expand(num_samples, -1).contiguous()
        else:
            ids = ids.contiguous()
            attention_mask = attention_mask.contiguous()

        total_batch = ids.size(0)
        finished = torch.zeros(total_batch, dtype=torch.bool, device=ids.device)
        past_key_values = None
        concepts = None

        row_intervene = None
        row_apply_mask = None
        if interventions is not None:
            row_intervene = interventions.to(device=ids.device, dtype=torch.float32)
            if row_intervene.size(0) == prompt_batch and num_samples > 1:
                row_intervene = row_intervene.repeat_interleave(num_samples, dim=0)
        if intervention_mask is not None:
            row_apply_mask = intervention_mask.to(device=ids.device, dtype=torch.bool).view(-1)
            if row_apply_mask.numel() == prompt_batch and num_samples > 1:
                row_apply_mask = row_apply_mask.repeat_interleave(num_samples, dim=0)

        def _intervene_fn(concepts):
            if row_intervene is None:
                return concepts
            iv = row_intervene.to(device=concepts.device, dtype=concepts.dtype).unsqueeze(1).expand_as(concepts)
            if row_apply_mask is None:
                apply_mask = torch.ones((total_batch, 1, 1), dtype=torch.bool, device=concepts.device).expand_as(concepts)
            else:
                apply_mask = row_apply_mask.view(-1, 1, 1).expand_as(concepts)
            if keep_other_concepts:
                apply_mask = apply_mask & (row_intervene.to(device=concepts.device) != 0).unsqueeze(1).expand_as(concepts)
            return torch.where(apply_mask, iv, concepts)

        for _ in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                intervene_concepts_fn=_intervene_fn if row_intervene is not None else None,
                llama_vocab_weight=llama_vocab_weight,
            )
            for b in range(total_batch):
                if not finished[b]:
                    token_mask = attention_mask[b].bool()
                    if token_mask.any():
                        token_ids = ids[b][token_mask]
                        score = logits[b, -1, token_ids].clone()
                        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                        logits[b, -1, token_ids] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return ids, concepts if concepts is not None else None

    def generate_multi_concept_batch(
        self,
        ids,
        preLM,
        interventions,
        samples_per_intervention=1,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """
        Generate samples for multiple concept interventions in a single batch.

        Output rows are grouped by intervention:
          [interv_0_sample_0, ..., interv_0_sample_{n-1},
           interv_1_sample_0, ..., interv_{K-1}_sample_{n-1}]

        Args:
            ids: (1, prompt_len) input token ids (will be broadcast).
            interventions: list of K intervention vectors, each of length concept_dim.
            samples_per_intervention: how many samples to generate per intervention.

        Returns:
            ids: (K * samples_per_intervention, seq_len) generated token ids.
            concepts: final activated concepts tensor, or None.
        """
        num_groups = len(interventions)
        total_batch = num_groups * samples_per_intervention

        ids = ids.expand(total_batch, -1).contiguous()
        finished = torch.zeros(total_batch, dtype=torch.bool, device=ids.device)

        intervention_tensor = torch.tensor(
            interventions, dtype=torch.float32, device=ids.device
        )  # (K, concept_dim)
        intervention_expanded = intervention_tensor.repeat_interleave(
            samples_per_intervention, dim=0
        )  # (total_batch, concept_dim)

        past_key_values = None
        concepts = None

        def _intervene_fn(concepts):
            iv = intervention_expanded.unsqueeze(1).expand_as(concepts)
            if not keep_other_concepts:
                return iv.contiguous()
            mask = (intervention_expanded != 0).unsqueeze(1).expand_as(concepts)
            return torch.where(mask, iv, concepts)

        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                intervene_concepts_fn=_intervene_fn,
                llama_vocab_weight=llama_vocab_weight,
            )
            for b in range(total_batch):
                if not finished[b]:
                    score = logits[b, -1, ids[b]].clone()
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, -1, ids[b]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return ids, concepts if concepts is not None else None


class CBLResidual(nn.Module):
    def __init__(self, config, concept_dim, residual_dim, tokenizer, cbl_layer_idx: int = -1, use_residual: bool = True):
        super().__init__()
        self.cbl = nn.Linear(config.hidden_size, concept_dim)
        self.cbl_residual = nn.Linear(config.hidden_size, residual_dim)
        self.fc = nn.Linear(concept_dim + residual_dim, config.vocab_size)
        self.relu = nn.ReLU()
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim
        self.tokenizer = tokenizer
        self.match_layer = None
        if concept_dim != residual_dim:
            print("Warning: concept_dim and residual_dim are not equal so creating a linear layer to match dimensions.")
            self.match_layer = nn.Linear(residual_dim, concept_dim)
        self.cbl_layer_idx = int(cbl_layer_idx)
        self.use_residual = bool(use_residual)
        self.hidden_size = config.hidden_size
        self.proj = None
        if self.cbl_layer_idx >= 0:
            self.proj = nn.Linear(concept_dim + residual_dim, config.hidden_size)

    @property
    def _unsup_branch(self):
        return self.cbl_residual

    def forward(self, features, llama_logits=None):
        concepts = self.cbl(features)
        unsup_features = self.cbl_residual(features)
        # print("concepts shape:", concepts.shape)
        # print("unsup_features shape:", unsup_features.shape)
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return self.relu(concepts), unsup_features, logits, self.match_layer(unsup_features) if self.match_layer else unsup_features

    def forward_full(self, preLM, input_ids, attention_mask=None, llama_vocab_weight=None):
        """Run the full backbone + CBL/bottleneck. See ``CBL.forward_full``."""
        if self.cbl_layer_idx == -1:
            features = preLM(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            llama_logits = (
                F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            )
            concepts, unsup, vocabs, matched = self.forward(features.float(), llama_logits=llama_logits)
            return concepts, unsup, vocabs, matched, None, None
        if llama_vocab_weight is None:
            raise ValueError("llama_vocab_weight is required when cbl_layer_idx >= 0.")
        llama_model = _get_llama_model(preLM)
        target_layer = llama_model.layers[self.cbl_layer_idx]
        store: dict = {}

        def _hook(module, args, kwargs, output):
            h_L = output[0] if isinstance(output, tuple) else output
            feats = h_L.float()
            concepts_raw = self.cbl(feats)
            unsup = self.cbl_residual(feats)
            concepts_relu = self.relu(concepts_raw)
            h_L_proj_f = self.proj(torch.cat((concepts_relu, unsup), dim=-1))
            h_L_proj = h_L_proj_f.to(h_L.dtype)
            if self.use_residual:
                h_L_out = h_L_proj + h_L
            else:
                h_L_out = h_L_proj
            store["concepts"] = concepts_relu
            store["unsup"] = unsup
            store["h_L"] = h_L
            store["h_L_proj"] = h_L_proj_f
            if isinstance(output, tuple):
                return (h_L_out,) + tuple(output[1:])
            return h_L_out

        handle = target_layer.register_forward_hook(_hook, with_kwargs=True)
        try:
            outputs = preLM(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        h_final = outputs.last_hidden_state
        vocabs = F.linear(h_final.to(llama_vocab_weight.dtype), llama_vocab_weight)
        unsup = store["unsup"]
        matched = self.match_layer(unsup) if self.match_layer else unsup
        return store["concepts"], unsup, vocabs, matched, store["h_L"], store["h_L_proj"]

    def intervene(self, unsup_features, intervene, llama_logits=None):
        concepts = intervene
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits

    def intervene_full(self, preLM, h_L, attention_mask, intervened_concepts, unsup_features, llama_vocab_weight):
        """Tail-only re-forward for intervention loss in intermediate mode."""
        if self.cbl_layer_idx == -1:
            raise RuntimeError("intervene_full is only valid in intermediate mode (cbl_layer_idx >= 0).")
        h_L_proj_f = self.proj(torch.cat((self.relu(intervened_concepts), unsup_features), dim=-1))
        h_L_proj = h_L_proj_f.to(h_L.dtype)
        h_L_out = (h_L_proj + h_L) if self.use_residual else h_L_proj
        llama_model = _get_llama_model(preLM)
        h_final = _bottleneck_forward_tail(llama_model, h_L_out, attention_mask, self.cbl_layer_idx)
        return F.linear(h_final.to(llama_vocab_weight.dtype), llama_vocab_weight)


    def generate(self, ids, preLM, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001, llama_vocab_weight=None):
        past_key_values = None

        def _intervene_fn(concepts):
            if intervene:
                for j in range(self.concept_dim):
                    concepts[0, :, j] = intervene[j]
            return concepts

        concepts = None
        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                intervene_concepts_fn=_intervene_fn if intervene else None,
                llama_vocab_weight=llama_vocab_weight,
            )
            score = logits[:, -1, ids[0]]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[:, -1, ids[0]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return ids, concepts[0] if concepts is not None else None

    def generate_batch(
        self,
        ids,
        preLM,
        num_samples=1,
        intervene=None,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """Generate num_samples trajectories in parallel (batched autoregressive)."""
        ids = ids.expand(num_samples, -1).contiguous()  # (B, prompt_len)
        finished = torch.zeros(num_samples, dtype=torch.bool, device=ids.device)
        past_key_values = None
        concepts = None

        def _intervene_fn(concepts):
            if intervene and not keep_other_concepts:
                for j in range(self.concept_dim):
                    concepts[:, :, j] = intervene[j]
            elif intervene and keep_other_concepts:
                for j in range(self.concept_dim):
                    val = intervene[j]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if val != 0:
                        concepts[:, :, j] = val
            return concepts

        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            logits, concepts, past_key_values, _ = _bottleneck_step_logits(
                self,
                preLM,
                input_ids,
                past_key_values=past_key_values,
                intervene_concepts_fn=_intervene_fn if intervene else None,
                llama_vocab_weight=llama_vocab_weight,
            )
            for b in range(num_samples):
                if not finished[b]:
                    score = logits[b, -1, ids[b]].clone()
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, -1, ids[b]] = score
            next_token_logits = logits[:, -1, :] / temp  # (B, vocab_size)
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)  # (B, 1)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return ids, concepts if concepts is not None else None

    def generate_intervention_batch_parallel(
        self,
        ids,
        preLM,
        attention_mask,
        num_samples=1,
        interventions=None,
        intervention_mask=None,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """Generate for batched prompts in parallel with optional per-row interventions."""
        prompt_batch = ids.size(0)
        if prompt_batch > 1 and num_samples > 1:
            ids = ids.repeat_interleave(num_samples, dim=0).contiguous()
            attention_mask = attention_mask.repeat_interleave(num_samples, dim=0).contiguous()
        elif prompt_batch == 1 and num_samples > 1:
            ids = ids.expand(num_samples, -1).contiguous()
            attention_mask = attention_mask.expand(num_samples, -1).contiguous()
        else:
            ids = ids.contiguous()
            attention_mask = attention_mask.contiguous()

        total_batch = ids.size(0)
        finished = torch.zeros(total_batch, dtype=torch.bool, device=ids.device)
        past_key_values = None
        concepts = None

        row_intervene = None
        row_apply_mask = None
        if interventions is not None:
            row_intervene = interventions.to(device=ids.device, dtype=torch.float32)
            if row_intervene.size(0) == prompt_batch and num_samples > 1:
                row_intervene = row_intervene.repeat_interleave(num_samples, dim=0)
        if intervention_mask is not None:
            row_apply_mask = intervention_mask.to(device=ids.device, dtype=torch.bool).view(-1)
            if row_apply_mask.numel() == prompt_batch and num_samples > 1:
                row_apply_mask = row_apply_mask.repeat_interleave(num_samples, dim=0)

        for _ in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            outputs = preLM(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.cbl_residual(features)

            if row_intervene is not None:
                iv = row_intervene.to(device=concepts.device, dtype=concepts.dtype).unsqueeze(1).expand_as(concepts)
                if row_apply_mask is None:
                    apply_mask = torch.ones((total_batch, 1, 1), dtype=torch.bool, device=concepts.device).expand_as(concepts)
                else:
                    apply_mask = row_apply_mask.view(-1, 1, 1).expand_as(concepts)
                if keep_other_concepts:
                    apply_mask = apply_mask & (row_intervene.to(device=concepts.device) != 0).unsqueeze(1).expand_as(concepts)
                concepts = torch.where(apply_mask, iv, concepts)

            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            if llama_vocab_weight is not None:
                llama_logits = F.linear(outputs.last_hidden_state.to(llama_vocab_weight.dtype), llama_vocab_weight)
                logits = logits + llama_logits.to(dtype=logits.dtype)
            for b in range(total_batch):
                if not finished[b]:
                    token_mask = attention_mask[b].bool()
                    if token_mask.any():
                        token_ids = ids[b][token_mask]
                        score = logits[b, -1, token_ids].clone()
                        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                        logits[b, -1, token_ids] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return ids, self.relu(concepts) if concepts is not None else None

    def generate_multi_concept_batch(
        self,
        ids,
        preLM,
        interventions,
        samples_per_intervention=1,
        length=100,
        temp=0.7,
        topk=100,
        topp=0.9,
        repetition_penalty=1.5,
        eos_token_id=128001,
        keep_other_concepts: bool = False,
        llama_vocab_weight=None,
    ):
        """
        Generate samples for multiple concept interventions in a single batch.

        Output rows are grouped by intervention:
          [interv_0_sample_0, ..., interv_0_sample_{n-1},
           interv_1_sample_0, ..., interv_{K-1}_sample_{n-1}]

        Args:
            ids: (1, prompt_len) input token ids (will be broadcast).
            interventions: list of K intervention vectors, each of length concept_dim.
            samples_per_intervention: how many samples to generate per intervention.

        Returns:
            ids: (K * samples_per_intervention, seq_len) generated token ids.
            concepts: final activated concepts tensor, or None.
        """
        num_groups = len(interventions)
        total_batch = num_groups * samples_per_intervention

        ids = ids.expand(total_batch, -1).contiguous()
        finished = torch.zeros(total_batch, dtype=torch.bool, device=ids.device)

        intervention_tensor = torch.tensor(
            interventions, dtype=torch.float32, device=ids.device
        )  # (K, concept_dim)
        intervention_expanded = intervention_tensor.repeat_interleave(
            samples_per_intervention, dim=0
        )  # (total_batch, concept_dim)

        past_key_values = None
        concepts = None

        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            outputs = preLM(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.cbl_residual(features)

            iv = intervention_expanded.unsqueeze(1).expand_as(concepts)
            if not keep_other_concepts:
                concepts = iv.contiguous()
            else:
                mask = (intervention_expanded != 0).unsqueeze(1).expand_as(concepts)
                concepts = torch.where(mask, iv, concepts)

            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            if llama_vocab_weight is not None:
                llama_logits = F.linear(
                    outputs.last_hidden_state.to(llama_vocab_weight.dtype),
                    llama_vocab_weight,
                )
                logits = logits + llama_logits.to(dtype=logits.dtype)
            for b in range(total_batch):
                if not finished[b]:
                    score = logits[b, -1, ids[b]].clone()
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, -1, ids[b]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering_batched(next_token_logits.clone(), top_k=topk, top_p=topp)
            next_token = _safe_multinomial_from_logits(filtered_logits)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return ids, self.relu(concepts) if concepts is not None else None

    def compute_residual_contrib(self, unsup_features):
        w = self.fc.weight  # shape: (vocab_size, concept_dim + residual_dim)
        # print("fc weight shape:", w.shape)
        w_non_concept = w[:, self.concept_dim:]  # shape: (vocab_size, residual_dim)
        # print("w_non_concept shape:", w_non_concept.shape)
        contrib = F.linear(unsup_features, w_non_concept)  # shape: (batch_size, vocab_size)
        # print("residual contrib shape:", contrib.shape)
        return contrib