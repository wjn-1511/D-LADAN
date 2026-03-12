import argparse
import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Config.parser import ConfigParser
from Model.DLADAN_model_torch import DLADANTorch
from utils.training_setup import setup_seed


class CAILTorchDataset(Dataset):
    def __init__(self, data_path: str, split: str, data_version: str):
        if data_version == "small":
            p = os.path.join(data_path, "normal", "small", f"{split}_processed_thulac.pkl")
        else:
            fname = f"{split}_processed_thulac_large.pkl" if split != "valid" else "test_processed_thulac_large.pkl"
            p = os.path.join(data_path, "normal", "large", fname)

        with open(p, "rb") as f:
            data = pkl.load(f)

        self.facts = np.asarray(data["fact_list"], dtype=np.int64)
        self.law = np.asarray(data["law_label_lists"], dtype=np.int64)
        self.accu = np.asarray(data["accu_label_lists"], dtype=np.int64)
        self.time = np.asarray(data["term_lists"], dtype=np.int64)

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        return {
            "fact": torch.tensor(self.facts[idx], dtype=torch.long),
            "law": torch.tensor(self.law[idx], dtype=torch.long),
            "accu": torch.tensor(self.accu[idx], dtype=torch.long),
            "time": torch.tensor(self.time[idx], dtype=torch.long),
        }


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, n = 0.0, 0
    correct = {"law": 0, "accu": 0, "time": 0}

    for batch in loader:
        fact = batch["fact"].to(device)
        labels = {k: batch[k].to(device) for k in ["law", "accu", "time"]}

        with torch.set_grad_enabled(train_mode):
            outputs = model(fact)
            loss = model.compute_loss(outputs, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bsz = fact.size(0)
        total_loss += loss.item() * bsz
        n += bsz

        correct["law"] += (outputs["law_logits"].argmax(dim=-1) == labels["law"]).sum().item()
        correct["accu"] += (outputs["accu_logits"].argmax(dim=-1) == labels["accu"]).sum().item()
        correct["time"] += (outputs["time_logits"].argmax(dim=-1) == labels["time"]).sum().item()

    metrics = {f"{k}_acc": correct[k] / max(1, n) for k in correct}
    metrics["loss"] = total_loss / max(1, n)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--data_path", type=str, default="../processed_dataset/CAIL")
    parser.add_argument("--data_version", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_npy", type=str, default="../data/cail_thulac.npy")
    parser.add_argument("--dry_run", action="store_true", help="Run one batch for migration sanity-check.")
    args = parser.parse_args()

    setup_seed(666)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu != "" and torch.cuda.is_available() else "cpu")

    config = ConfigParser("../Config/LadanPPK.config")
    embedding_dim = config.getint("data", "vec_size")
    hidden_size = config.getint("net", "han_size")

    if args.data_version == "small":
        law_num, accu_num, time_num = 103, 119, 11
    else:
        law_num, accu_num, time_num = 118, 130, 11

    train_set = CAILTorchDataset(args.data_path, "train", args.data_version)
    valid_set = CAILTorchDataset(args.data_path, "valid", args.data_version)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    emb = np.load(args.embedding_npy)
    vocab_size = emb.shape[0]

    model = DLADANTorch(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        group_num=law_num,
        law_num=law_num,
        accu_num=accu_num,
        time_num=time_num,
        embedding_matrix=emb,
        embedding_trainable=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.dry_run:
        one_batch = [next(iter(train_loader))]
        metrics = run_epoch(model, one_batch, device, optimizer=optimizer)
        print("dry_run:", metrics)
        return

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        valid_metrics = run_epoch(model, valid_loader, device, optimizer=None)
        print(f"epoch={epoch + 1} train={train_metrics} valid={valid_metrics}")


if __name__ == "__main__":
    main()
