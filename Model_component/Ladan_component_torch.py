import torch
import torch.nn as nn


class LadanPPKTorch(nn.Module):
    """A lightweight PyTorch migration of the LadanPPK feature extractor.

    This module keeps the same high-level behavior (hierarchical encoding + group prediction)
    while removing TensorFlow/Keras dependencies so users can modify it more easily.
    """

    def __init__(self, vocab_dim: int, hidden_size: int, group_num: int, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.group_num = group_num

        self.word_encoder = nn.GRU(
            input_size=vocab_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.sent_encoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.group_classifier = nn.Linear(hidden_size * 2, group_num)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.float()
        denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom

    def forward(self, fact_desc: torch.Tensor, word_mask: torch.Tensor, sentence_mask: torch.Tensor):
        """
        Args:
            fact_desc: [B, S, W, E]
            word_mask: [B, S, W]
            sentence_mask: [B, S]
        Returns:
            fact_rep: [B, 2H]
            group_prob: [B, G]
        """
        bsz, sent_num, word_num, emb_dim = fact_desc.shape

        word_in = fact_desc.reshape(bsz * sent_num, word_num, emb_dim)
        word_mask_flat = word_mask.reshape(bsz * sent_num, word_num)

        word_out, _ = self.word_encoder(word_in)
        sent_rep = self.masked_mean(word_out, word_mask_flat, dim=1)
        sent_rep = sent_rep.reshape(bsz, sent_num, -1)

        sent_out, _ = self.sent_encoder(sent_rep)
        doc_rep = self.masked_mean(sent_out, sentence_mask, dim=1)
        doc_rep = self.dropout(doc_rep)

        group_logits = self.group_classifier(doc_rep)
        group_prob = torch.softmax(group_logits, dim=-1)
        return doc_rep, group_prob
