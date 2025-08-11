# train_local.py
import argparse
import os
import json
import torch
import numpy as np
from model import Model
from main import run_trainval
from typing import Optional

def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_init_weights_if_any(model: torch.nn.Module, init_model_path: Optional[str]):
    if not init_model_path:
        return
    ckpt = torch.load(init_model_path, map_location="cpu")
    # 兼容两种常见保存格式
    state = ckpt.get("xin_graph_seq2seq_model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded init weights from {init_model_path}")

def save_edge_importance(model: Model, out_edge_path: str):
    with torch.no_grad():
        edge_importance = [p.detach().cpu().numpy() for p in model.edge_importance]
    ensure_parent(out_edge_path)
    np.save(out_edge_path, edge_importance)
    print(f"[SAVE] edge importance -> {out_edge_path}")

def save_model(model: Model, out_model_path: str):
    ensure_parent(out_model_path)
    torch.save({"xin_graph_seq2seq_model": model.state_dict()}, out_model_path)
    print(f"[SAVE] model -> {out_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a local GNN client with flexible I/O paths.")
    parser.add_argument("--train_data", required=True, help="Path to train_data_*.pkl")
    parser.add_argument("--test_data", default="processed_data/test_data.pkl", help="Path to test_data.pkl")
    parser.add_argument("--init_model", default=None, help="Optional path to init/warm-start model.pt")
    parser.add_argument("--out_model", required=True, help="Where to save trained local model .pt")
    parser.add_argument("--out_edge", required=True, help="Where to save edge importance .npy")
    parser.add_argument("--out_metrics", default=None, help="Optional: save a JSON with test metrics")
    # 下面两个通常不改，保持与现有工程一致
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--num_node", type=int, default=120)
    parser.add_argument("--max_hop", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_args = {"max_hop": args.max_hop, "num_node": args.num_node}

    model = Model(in_channels=args.in_channels, graph_args=graph_args, edge_importance_weighting=True).to(device)
    load_init_weights_if_any(model, args.init_model)

    # 训练 + 验证（保持与你的 main.run_trainval 一致的签名）
    # 如果 run_trainval 返回了测试指标，就记录下来；若无返回，也不报错
    maybe_metrics = run_trainval(
        model,
        pra_traindata_path=args.train_data,
        pra_testdata_path=args.test_data
    )

    # 保存产物
    save_edge_importance(model, args.out_edge)
    save_model(model, args.out_model)

    # 记录指标（可选）
    if args.out_metrics is not None:
        ensure_parent(args.out_metrics)
        payload = {}
        # 兼容 tuple / dict / None
        if maybe_metrics is None:
            payload["note"] = "run_trainval did not return metrics."
        elif isinstance(maybe_metrics, dict):
            payload = maybe_metrics
        else:
            payload["metrics"] = maybe_metrics
        with open(args.out_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVE] metrics -> {args.out_metrics}")

if __name__ == "__main__":
    main()
