from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]

            # 各パラメタ tensorごとにpを更新
            for p in group["params"]: 
                if p.grad is None:
                    continue

                # この勾配を使ってpを更新する！
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # Update first and second moments of the gradients

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # Update parameters

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

                # State should be stored in this dictionary
                # optimizer自体が管理する、各パラメタごとの状態変数を保存する辞書
                state = self.state[p]

                # State len check
                if len(state) == 0:
                    state['step'] = 0
                    # m (1st moment vector): 勾配の指数移動平均
                    state['exp_avg'] = torch.zeros_like(p)
                    # v (2nd moment vector): 勾配の二乗の指数移動平均
                    state['exp_avg_sq'] = torch.zeros_like(p)

                # 3. ハイパーパラメータとStateの取得
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 4. ステップの更新
                state['step'] += 1

                with torch.no_grad():
                    # A. Weight Decay (L2 Regularization) - Decoupled
                    # AdamWの特徴：勾配に基づく更新とは独立して、減衰させる。
                    # p = p - lr * lambda * p
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # B. モーメンタム (m) の更新
                    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # 二乗勾配の移動平均 (v) の更新
                    # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ^ 2)
                    # addcmul_ は (tensor + value * tensor1 * tensor2) を行う
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # バイアス補正項の計算
                    # 初期ステップでは m, v が0に偏るため、それを補正する係数
                    denom = exp_avg_sq.sqrt().add_(eps)

                    if correct_bias:
                        bc1 = 1 - beta1 ** state["step"]
                        bc2 = 1 - beta2 ** state["step"]

                        # バイアス補正を考慮したステップサイズを計算
                        step_size = lr * (bc2 ** 0.5) / bc1
                    else:
                        step_size = lr
                    
                    # パラメータの更新（p -= step_size * (m / (sqrt(v) + eps))）
                    
                    # addcdiv_ は (tensor + value * (tensor1 / tensor2)) を行う
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss