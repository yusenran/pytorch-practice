import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm

DEVICE = 'cpu'

# 減衰正弦波オシレーター
def exp_decay_sinusoid(z_omega : Tensor, z_phi :Tensor, n : int) -> Tensor:
    arr_n = torch.arange(n).to(DEVICE)
    magnitude = z_omega.abs().pow(arr_n) * z_phi.abs()
    sinusoid = torch.cos(arr_n*z_omega.angle() + z_phi.angle())
    return magnitude * sinusoid

def create_target_sinusoid(signal_length :int) -> Tensor:
    # 目標正弦波を作成
    # 周波数・振幅・初期位相ともにランダム
    target_freq = torch.rand(1).to(DEVICE)*torch.pi
    target_mag = torch.rand(1).to(DEVICE)
    target_phase = torch.rand(1).to(DEVICE)*torch.pi
    target_sinusoid = torch.cos(torch.arange(signal_length).to(DEVICE) * target_freq + target_phase) * target_mag

    print(f"Target freq/mag/phase: {target_freq.item():.4f} / {target_mag.item():.4f} / {target_phase.item():.4f}")
    return target_sinusoid.to(DEVICE)

def draw2compare(target : Tensor, estimated : Tensor):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111, xlabel="xlabel", ylabel='ylabel')

    ax.plot(target, label="target")
    ax.plot(estimated, label="estimated")

    ax.legend()  # 凡例表示
    plt.show()

def create_target_wave(signal_length :int, sinusoid_num :int) -> Tensor:
    target_wave = create_target_sinusoid(signal_length)
    for _ in range(sinusoid_num-1):
        target_wave += create_target_sinusoid(signal_length)
    return target_wave.to(DEVICE)

def create_estimated_wave(omega_list :list[Tensor], phi_list :list[Tensor], length:int, num:int) -> Tensor:
    estimated_sinusoid = exp_decay_sinusoid(omega_list[0], phi_list[0], length)
    for i in range(num-1):
        estimated_sinusoid += exp_decay_sinusoid(omega_list[i+1], phi_list[i+1], length)
    return estimated_sinusoid

def estimate_freq(target_wave :Tensor, signal_length :int, sinusoid_num :int, opt_step :int=50000) -> Tensor:
    # 複素数パラメータ初期化
    # 単位円上の適当な値で初期化します。
    omega_list: list[Tensor] = []
    phi_list: list[Tensor] = []

    for _ in range(sinusoid_num):
        init_freq = torch.rand(1).to(DEVICE)*torch.pi
        init_phase = torch.rand(1).to(DEVICE)*torch.pi
        omega = torch.exp(init_freq*1.0j).requires_grad_().to(DEVICE)    # 角周波数・減衰係数パラメーター
        phi = torch.exp(init_phase*1.0j).requires_grad_().to(DEVICE)     # 振幅・初期位相パラメーター
        omega_list.append(omega.to(DEVICE))
        phi_list.append(phi.to(DEVICE))
        # 初期パラメーターをプリント
        print(f"Initial freq/mag/phase: {omega.angle().item():.4f} / {phi.abs().item():.4f} / {phi.angle().item():.4f}")

    # 勾配降下ループ
    optimizer = torch.optim.Adam(omega_list + phi_list, lr=1e-4)
    for _ in tqdm.tqdm(range(opt_step),desc="Gradient descent"):
        # パラメーターから減衰正弦波を生成
        estimated_wave = create_estimated_wave(omega_list, phi_list, signal_length, sinusoid_num)
        # 損失関数：Mean Squared Error
        loss = F.mse_loss(estimated_wave.to(DEVICE), target_wave.to(DEVICE)).to(DEVICE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-4:
            break

    print(f"Loss: {loss.item()}")
    for i in range(sinusoid_num):
        print(f"Estimated freq/mag/phase: {omega_list[i].angle().item():.4f} / {phi_list[i].abs().item():.4f} / {phi_list[i].angle().item():.4f}")
    estimated_wave = create_estimated_wave(omega_list, phi_list, signal_length, sinusoid_num)
    return estimated_wave.detach()

if __name__ == "__main__":
    signal_length = 4096
    sinusoid_num = 2
    target_wave = create_target_wave(signal_length, sinusoid_num).cpu()
    estimated_wave = estimate_freq(target_wave, signal_length, sinusoid_num).cpu()
    draw2compare(target_wave[:300], estimated_wave[:300])