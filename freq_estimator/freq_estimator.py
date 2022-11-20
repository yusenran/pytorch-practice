# ライブラリ読込
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm

# 減衰正弦波オシレーター
def exp_decay_sinusoid(z_omega : Tensor, z_phi :Tensor, n : int) -> Tensor:
    arr_n = torch.arange(n)
    magnitude = z_omega.abs().pow(arr_n) * z_phi.abs()
    sinusoid = torch.cos(arr_n*z_omega.angle() + z_phi.angle())
    return magnitude * sinusoid

def estimate_freq():
    # 複素数パラメータ初期化
    # 単位円上の適当な値で初期化します。
    init_freq = torch.rand(1)*torch.pi
    init_phase = torch.rand(1)*torch.pi
    param_omega = torch.exp(init_freq*1.0j).requires_grad_()    # 角周波数・減衰係数パラメーター
    param_phi = torch.exp(init_phase*1.0j).requires_grad_()     # 振幅・初期位相パラメーター

    # 目標正弦波を作成
    # 長さ4096、周波数・振幅・初期位相ともにランダム
    signal_length=4096
    target_freq = torch.rand(1)*torch.pi
    target_mag = torch.rand(1)
    target_phase = torch.rand(1)*torch.pi
    target_sinusoid = torch.cos(torch.arange(signal_length) * target_freq + target_phase) * target_mag

    # 初期・目標パラメーターをプリント
    print(f"Initial freq/mag/phase: {param_omega.angle().item():.4f} / {param_phi.abs().item():.4f} / {param_phi.angle().item():.4f}")
    print(f"Target freq/mag/phase: {target_freq.item():.4f} / {target_mag.item():.4f} / {target_phase.item():.4f}")

    # 勾配降下ループ
    optimizer = torch.optim.Adam([param_omega, param_phi], lr=1e-4)
    for _ in tqdm.tqdm(range(50000),desc="Gradient descent"):
        # パラメーターから減衰正弦波を生成
        estimated_sinusoid = exp_decay_sinusoid(param_omega, param_phi, signal_length)
        # 損失関数：Mean Squared Error
        loss = F.mse_loss(estimated_sinusoid, target_sinusoid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")
    print(f"Estimated freq/mag/phase: {param_omega.angle().item():.4f} / {param_phi.abs().item():.4f} / {param_phi.angle().item():.4f}")

if __name__ == "__main__":
    estimate_freq()