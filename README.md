# GNN-based Reinforcement Learning for Robust TSN Routing

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

그래프 어텐션 네트워크(GAT)와 강화학습을 이용해 시분할 حساس 네트워크(TSN)의 견고한(Robust) 라우팅 경로를 찾는 프로젝트입니다.

## 🌟 Key Features

-   **Advanced GNN Model**: `GATv2`를 사용하여 네트워크의 위상적 특징과 노드 간의 중요도를 학습합니다.
-   **Hybrid Learning Strategy**: ILP(정수 선형 계획법) 솔버를 '앵커'로 사용하는 독창적인 강화학습 프레임워크를 통해 GNN이 전문가의 성능을 뛰어넘도록 학습합니다.
-   **Rich State Representation**: 단순한 노드 정보가 아닌, 네트워크의 혼잡도(congestion)와 플로우의 대역폭 요구사항 등 풍부한 상태 정보를 GNN에 제공하여 지능적인 의사결정을 유도합니다.
-   **Robustness Evaluation**: 정상 상태뿐만 아니라, 링크 고장(link failure) 시나리오를 포함하여 실제 환경과 유사한 견고성을 평가합니다.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/GNN-TSN-Routing.git](https://github.com/your-username/GNN-TSN-Routing.git)
    cd GNN-TSN-Routing
    ```

2.  **Create & Activate Conda Environment:**
    ```bash
    conda create -n tsn_env python=3.9 -y
    conda activate tsn_env
    ```

3.  **Install Dependencies:**
    PyTorch를 CUDA 버전에 맞게 먼저 설치 후, 나머지 라이브러리를 설치합니다.
    ```bash
    # For CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Install other packages
    pip install -r requirements.txt
    
    # Install ILP Solver (for Debian/Ubuntu)
    sudo apt-get install -y coinor-cbc
    ```

## 🚀 Usage

**1. Train a new model:**
전체 학습 파이프라인(모방학습 -> 온라인 강화학습 -> 벤치마크)을 실행합니다.
```bash
python main.py --mode train
```

**2. Run benchmark with a pre-trained model:**
(기능 추가 필요) 학습된 모델을 사용하여 벤치마크만 실행합니다.
```bash
python main.py --mode benchmark --model_path data/gnn_ultimate_v3.pth
```

## 📈 Methodology

본 프로젝트는 2단계 학습 전략을 사용합니다.

1.  **Phase 1: Imitation Learning (Warm-up)**: GNN 에이전트가 `Greedy` 알고리즘의 라우팅 결과를 모방하여 기본적인 경로 탐색 능력을 빠르게 학습합니다.
2.  **Phase 2: ILP-Anchored Online Learning**: GNN 에이전트가 `ILP Solver`와 경쟁하며 학습을 진행합니다. GNN의 해가 ILP보다 우수하면 강화학습(Policy Gradient)을 통해 정책을 강화하고, ILP의 해가 더 우수하면 모방학습(Behavioral Cloning)을 통해 정책을 교정합니다.

## 📊 Results

학습 완료 후, `data/` 폴더에 GNN, Greedy, ILP 솔버의 성능(점수)과 계산 시간을 비교한 아래와 같은 그래프가 생성됩니다.

*(여기에 최종 결과 그래프 이미지 삽입)*

| Solver  | Average Score | Average Time (s) |
| :------ | :------------ | :--------------- |
| GNN_v3  | 0.95+         | ~0.1s            |
| ILP     | ~0.90         | ~20s             |
| Greedy  | ~0.85         | <0.01s           |
*(결과 예시 테이블)*

## 📚 Future Work

-   Graph Transformer 모델 도입
-   Multi-Agent RL(MARL) 접근법 적용
-   더 복잡하고 다양한 TSN 스케줄링 알고리즘 통합

## 📄 License

This project is licensed under the MIT License.
