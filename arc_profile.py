"""诊断 ARC JEPA 训练的时间瓶颈"""
import time
import torch
from mvp_config import config
from arc_trainer import ArcJEPA, load_arc_tasks

model = ArcJEPA().to(config.device)
tasks = load_arc_tasks(10, "training")

# 统计各部分耗时
times = {"encode": 0, "predictor": 0, "decode": 0, "backward": 0, "total": 0}

for task in tasks:
    t0 = time.perf_counter()

    # Encode
    t_enc = time.perf_counter()
    context_seq, z_target, H, W = model.encode_task(task)
    torch.cuda.synchronize()
    times["encode"] += time.perf_counter() - t_enc

    T = context_seq.shape[1]
    intent = model.arc_intent.expand(-1, T, -1)

    # Predictor
    t_pred = time.perf_counter()
    pred_seq = model.predictor(context_seq, intent)
    torch.cuda.synchronize()
    times["predictor"] += time.perf_counter() - t_pred

    z_pred = pred_seq[:, -1, :]

    # Decode
    t_dec = time.perf_counter()
    logits = model.grid_decoder(z_pred, H, W)
    torch.cuda.synchronize()
    times["decode"] += time.perf_counter() - t_dec

    gold = torch.tensor(task["test"][0]["output"], dtype=torch.long, device=config.device)
    import torch.nn.functional as F
    loss = F.mse_loss(z_pred, z_target.detach()) + F.cross_entropy(
        logits.view(-1, 10), gold.view(-1))

    # Backward
    t_bw = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    times["backward"] += time.perf_counter() - t_bw

    times["total"] += time.perf_counter() - t0
    model.zero_grad()

n = len(tasks)
print(f"\n📊 ARC JEPA 性能诊断 ({n} 题)")
print(f"{'组件':<15} {'总耗时':<10} {'每题':<10} {'占比':<8}")
print("-" * 45)
for k in ["encode", "predictor", "decode", "backward"]:
    pct = times[k] / times["total"] * 100
    print(f"  {k:<13} {times[k]:.3f}s    {times[k]/n*1000:.0f}ms     {pct:.0f}%")
print(f"  {'TOTAL':<13} {times['total']:.3f}s    {times['total']/n*1000:.0f}ms")
print(f"\n  推算 400 题/epoch: {times['total']/n*400:.0f}s")

# 分析 grid 尺寸分布
sizes = []
for t in tasks:
    for p in t["train"]:
        sizes.append(len(p["input"]) * len(p["input"][0]))
    sizes.append(len(t["test"][0]["input"]) * len(t["test"][0]["input"][0]))
print(f"\n  Grid cell 数分布: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")
