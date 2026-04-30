"""
CBM modules and utilities for TruthfulQA.
Self-contained: CBL, CBLResidual, sampling utilities, loss helpers.
Adapted from CB-LLMs/generation/modules.py and utils.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=float("-inf")):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits


def top_k_top_p_filtering_batched(logits, top_k=0, top_p=0.0, filter_value=float("-inf")):
    if top_k > 0:
        top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0]
        logits = logits.masked_fill(logits < top_k_vals[:, -1:], filter_value)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_rm = cumulative_probs > top_p
        sorted_rm[..., 1:] = sorted_rm[..., :-1].clone()
        sorted_rm[..., 0] = 0
        rm = sorted_rm.scatter(1, sorted_indices, sorted_rm)
        logits = logits.masked_fill(rm, filter_value)
    return logits


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def elastic_net_penalty(param, alpha=0.99):
    return alpha * torch.abs(param).mean() + (1 - alpha) * torch.square(param).mean()


def cos_sim_cubed(features, target, reduce=True):
    features = features - features.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    features = F.normalize(features ** 3, dim=-1)
    target = F.normalize(target ** 3, dim=-1)
    sim = (features * target).sum(dim=-1)
    return sim.mean() if reduce else sim


def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


# ---------------------------------------------------------------------------
# CBL — Concept Bottleneck Layer
# ---------------------------------------------------------------------------

class CBL(nn.Module):
    def __init__(self, config, concept_dim, tokenizer):
        super().__init__()
        self.cbl = nn.Linear(config.hidden_size, concept_dim)
        self.unsup = nn.Linear(config.hidden_size, 768)
        self.fc = nn.Linear(concept_dim + 768, config.vocab_size)
        self.relu = nn.ReLU()
        self.concept_dim = concept_dim
        self.tokenizer = tokenizer

    def forward(self, features, llama_logits=None):
        concepts = self.cbl(features)
        unsup_features = self.unsup(features)
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return self.relu(concepts), unsup_features, logits, unsup_features

    def intervene(self, unsup_features, intervene, llama_logits=None):
        e = torch.cat((self.relu(intervene), unsup_features), dim=-1)
        logits = self.fc(e)
        if llama_logits is not None:
            logits = logits + llama_logits.to(dtype=logits.dtype)
        return logits

    def generate_batch(
        self, ids, preLM, num_samples=1, intervene=None, length=100,
        temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5,
        eos_token_id=128001,
    ):
        ids = ids.expand(num_samples, -1).contiguous()
        finished = torch.zeros(num_samples, dtype=torch.bool, device=ids.device)
        past_key_values = None
        concepts = None
        for _ in range(length):
            input_ids = ids[:, -1:] if past_key_values is not None else ids
            outputs = preLM(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            features = outputs.last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.unsup(features)
            if intervene:
                for j in range(self.concept_dim):
                    concepts[:, :, j] = intervene[j]
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            for b in range(num_samples):
                if not finished[b]:
                    score = logits[b, -1, ids[b]].clone()
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, -1, ids[b]] = score
            next_logits = logits[:, -1, :] / temp
            filtered = top_k_top_p_filtering_batched(next_logits.clone(), top_k=topk, top_p=topp)
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
            next_token[finished] = eos_token_id
            ids = torch.cat((ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return ids, self.relu(concepts) if concepts is not None else None
