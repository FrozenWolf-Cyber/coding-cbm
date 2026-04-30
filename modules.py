import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering, top_k_top_p_filtering_batched


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
    def __init__(self, config, concept_dim, tokenizer):
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

    def forward(self, features, llama_logits=None):
        concepts = self.cbl(features)
        unsup_features = self.unsup(features)
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return self.relu(concepts), unsup_features, logits, self.match_layer(unsup_features) if self.match_layer else unsup_features

    def intervene(self, unsup_features, intervene, llama_logits=None):
        concepts = intervene
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits

    def generate(self, ids, preLM, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001, llama_vocab_weight=None):
        past_key_values = None
        for i in range(length):
            outputs = preLM(ids[:, -1:] if past_key_values is not None else ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.unsup(features)
            if intervene:
                for j in range(self.concept_dim):
                    concepts[0, :, j] = intervene[j]
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
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
        return ids, self.relu(concepts)[0]

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
        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            outputs = preLM(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.unsup(features)

            # Intervention application (default behavior preserved):
            # - keep_other_concepts=False: overwrite *all* concept dims from `intervene` (including zeros)
            # - keep_other_concepts=True: overwrite only dims where `intervene[j] != 0`, leaving others as-is
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
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            if llama_vocab_weight is not None:
                llama_logits = F.linear(outputs.last_hidden_state.to(llama_vocab_weight.dtype), llama_vocab_weight)
                logits = logits + llama_logits.to(dtype=logits.dtype)
            # Per-sample repetition penalty
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
            unsup_features = self.unsup(features)

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


class CBLResidual(nn.Module):
    def __init__(self, config, concept_dim, residual_dim, tokenizer):
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

    def intervene(self, unsup_features, intervene, llama_logits=None):
        concepts = intervene
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits


    def generate(self, ids, preLM, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001, llama_vocab_weight=None):
        past_key_values = None
        for i in range(length):
            outputs = preLM(ids[:, -1:] if past_key_values is not None else ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.cbl_residual(features)
            if intervene:
                for j in range(self.concept_dim):
                    concepts[0, :, j] = intervene[j]
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
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
        return ids, self.relu(concepts)[0]

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
        for i in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            outputs = preLM(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.cbl_residual(features)

            # Intervention application (default behavior preserved):
            # - keep_other_concepts=False: overwrite *all* concept dims from `intervene` (including zeros)
            # - keep_other_concepts=True: overwrite only dims where `intervene[j] != 0`, leaving others as-is
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
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            if llama_vocab_weight is not None:
                llama_logits = F.linear(outputs.last_hidden_state.to(llama_vocab_weight.dtype), llama_vocab_weight)
                logits = logits + llama_logits.to(dtype=logits.dtype)
            # Per-sample repetition penalty
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