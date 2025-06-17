# ==================================================
# 1) 라이브러리 임포트
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam

pyro.set_rng_seed(0)

# ==================================================
# 2) 태양흑점 데이터 로드 및 전처리
DATASET_URL = "http://www.sidc.be/silso/DATA/SN_y_tot_V2.0.csv"
dset = pd.read_csv(DATASET_URL, sep=";", skiprows=1)

# 'SN' 컬럼(흑점 개수)을 정수형(long) 텐서로 변환
data = torch.tensor(dset["SN"].values, dtype=torch.long)  # dtype=torch.long
N = data.shape[0]  # 예: 150

# ==================================================
# 3) stick-breaking 가중치 함수
def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

# ==================================================
# 4) Poisson DPMM 모델 정의
def model(data):
    global T
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1.0, alpha))
    T = beta.shape[0] + 1

    with pyro.plate("lambda_plate", T):
        lmbda = pyro.sample("lambda", Gamma(3.0, 0.05))  # lmbda.shape = (T,)

    with pyro.plate("data", N):   # data.shape = (N,)
        z = pyro.sample("z", Categorical(mix_weights(beta)))  
        # z.shape = (N,),  lmbda[z].shape = (N,) 
        pyro.sample("obs", Poisson(lmbda[z]), obs=data)  
        # Poisson(lmbda[z]) 의 batch_shape=(N,) → obs=data의 shape=(N,)와 딱 맞음

# ==================================================
# 5) Mean‐field 가이드 정의
def guide(data):
    kappa = pyro.param(
        "kappa",
        lambda:   Uniform(0.0, 2.0).sample([T - 1]), 
        constraint=constraints.positive
    )
    tau_0 = pyro.param(
        "tau_0",
        lambda: Uniform(0.0, 5.0).sample([T]),
        constraint=constraints.positive
    )
    tau_1 = pyro.param(
        "tau_1",
        lambda: LogNormal(-1.0, 1.0).sample([T]),
        constraint=constraints.positive
    )
    phi = pyro.param(
        "phi",
        lambda: Dirichlet(torch.ones(T) / T).sample([N]), 
        constraint=constraints.simplex
    )

    with pyro.plate("beta_plate", T - 1):
        pyro.sample("beta", Beta(torch.ones(T - 1), kappa))

    with pyro.plate("lambda_plate", T):
        pyro.sample("lambda", Gamma(tau_0, tau_1))

    with pyro.plate("data", N):
        pyro.sample("z", Categorical(phi))

# ==================================================
# 6) SVI 설정 및 학습
T = 20            # truncation 개수
alpha = 1.0       # 디리클렛 하이퍼파라미터
n_iter = 2000

optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
elbo_values = []

def train(num_iterations):
    pyro.clear_param_store()
    for _ in tqdm(range(num_iterations)):
        elbo = svi.step(data)
        elbo_values.append(elbo)
    print(f"ELBO per datum: {elbo_values[-1] / N:.3f}")

train(n_iter)

# ==================================================
# 7) 사후 추정치(posterior point estimates) 추출
tau0_optimal = pyro.param("tau_0").detach()        # shape = (T,)
tau1_optimal = pyro.param("tau_1").detach()        # shape = (T,)
kappa_optimal = pyro.param("kappa").detach()       # shape = (T-1,)

# Posterior Gamma 평균 = τ0 / τ1  (각 클러스터별 λ̂ₖ)
Bayes_Rates = tau0_optimal / tau1_optimal  # (T,)

# Posterior stick-breaking weight 평균 → ŵₖ (shape=(T,))
Bayes_Weights = mix_weights( 1.0 / (1.0 + kappa_optimal) )  # (T,)

# ==================================================
# 8) Mixture of Poisson 분포 밀도 계산 & 플롯
samples = torch.arange(0, 300).type(torch.float)  # (300,)

def mixture_of_poisson(weights, rates, samples):
    # weights: (T,) → (1,T),  rates: (T,) → (1,T),  samples: (300,) → (300,1)
    w = weights.view(1, -1)       # (1, T)
    r = rates.view(1, -1)         # (1, T)
    x = samples.unsqueeze(-1)     # (300, 1)

    log_p = Poisson(r).log_prob(x)    # → shape (300, T) 
    prob  = log_p.exp()               # → shape (300, T)
    weighted = w * prob               # → shape (300, T)
    return weighted.sum(dim=1)        # → shape (300,)

likelihood = mixture_of_poisson(Bayes_Weights, Bayes_Rates, samples)

plt.figure(figsize=(8,4))
plt.title("Mixture of Poisson Distributions (Estimated)")
plt.xlabel("Sunspot Count")
plt.hist(
    data.numpy(),       # 관측된 카운트 데이터 (정수)
    bins=30, 
    density=True, 
    alpha=0.5, 
    label="Observed Data"
)
plt.plot(
    samples.numpy(), 
    likelihood.detach().numpy(), 
    color="red", 
    linewidth=2, 
    label="Estimated Mixture PMF"
)
plt.legend()
plt.show()
