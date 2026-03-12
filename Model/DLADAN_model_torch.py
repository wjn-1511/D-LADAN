import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_component.Ladan_component_torch import LadanPPKTorch


class DLADANTorch(nn.Module):
    """PyTorch version of DLADAN with multitask heads (law/accu/time + group).

    This implementation intentionally focuses on the core trainable path so later
    experimentation can be done in PyTorch without TensorFlow dependencies.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        group_num: int,
        law_num: int,
        accu_num: int,
        time_num: int,
        padding_idx: int = 0,
        dropout: float = 0.5,
        embedding_matrix=None,
        embedding_trainable: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if embedding_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = embedding_trainable

        self.feature_extractor = LadanPPKTorch(
            vocab_dim=embedding_dim,
            hidden_size=hidden_size,
            group_num=group_num,
            dropout=dropout,
        )

        self.law_head = nn.Linear(hidden_size * 2, law_num)
        self.accu_head = nn.Linear(hidden_size * 2, accu_num)
        self.time_head = nn.Linear(hidden_size * 2, time_num)

    def forward(self, fact_input: torch.Tensor):
        # fact_input: [B, S, W]
        word_mask = (fact_input != 0).float()
        sentence_mask = (word_mask.sum(dim=-1) > 0).float()

        fact_desc = self.embedding(fact_input)
        fact_rep, group_prob = self.feature_extractor(fact_desc, word_mask, sentence_mask)

        law_logits = self.law_head(fact_rep)
        accu_logits = self.accu_head(fact_rep)
        time_logits = self.time_head(fact_rep)

        return {
            "law_logits": law_logits,
            "accu_logits": accu_logits,
            "time_logits": time_logits,
            "group_prob": group_prob,
        }

    @staticmethod
    def compute_loss(outputs, labels, group_labels=None, group_weight: float = 0.1):
        loss_law = F.cross_entropy(outputs["law_logits"], labels["law"])
        loss_accu = F.cross_entropy(outputs["accu_logits"], labels["accu"])
        loss_time = F.cross_entropy(outputs["time_logits"], labels["time"])
        total = loss_law + loss_accu + loss_time

        if group_labels is not None:
            loss_group = F.nll_loss(torch.log(outputs["group_prob"].clamp_min(1e-12)), group_labels)
            total = total + group_weight * loss_group

        return total
