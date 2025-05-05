import torch

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None) -> dict:
    """
    计算深度估计的常用评估指标：AbsRel, RMSE, δ1, δ2, δ3

    参数：
        pred (torch.Tensor): 预测深度，形状 [B, H, W]
        gt   (torch.Tensor): 地面真值深度，形状 [B, H, W]
        mask (torch.Tensor): 可选的掩码，形状 [B, H, W]，True 表示有效像素

    返回：
        dict: 包含 'AbsRel', 'RMSE', 'δ1', 'δ2', 'δ3' 的平均值
    """
    if mask is None:
        mask = gt > 0
    # 仅考虑有效区域
    pred_vals = pred[mask]
    gt_vals = gt[mask]

    # 绝对相对误差
    abs_rel = torch.mean(torch.abs(pred_vals - gt_vals) / gt_vals).item()
    # 均方根误差
    rmse = torch.sqrt(torch.mean((pred_vals - gt_vals) ** 2)).item()
    # delta metrics
    ratio = torch.max(pred_vals / gt_vals, gt_vals / pred_vals)
    delta1 = torch.mean((ratio < 1.25).float()).item()
    delta2 = torch.mean((ratio < 1.25 ** 2).float()).item()
    delta3 = torch.mean((ratio < 1.25 ** 3).float()).item()

    return {
        'AbsRel': abs_rel,
        'RMSE': rmse,
        'δ1': delta1,
        'δ2': delta2,
        'δ3': delta3
    }
