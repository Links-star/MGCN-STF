# MGCN-STF: Multi-Graph Convolutional Network with Spatiotemporal Transformer Fusion  
## Project Overview  
MGCN-STF is a spatiotemporal sequence prediction model inspired by the ST-MGCN framework (from *Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting*). Designed for region-level tasks like ride-hailing demand forecasting, it focuses on capturing non-Euclidean spatial correlations and long-range temporal dependencies, with improvements in temporal modeling and attention mechanisms.  


## Background & Challenges  
Traditional models struggle with non-Euclidean spatial relationships (e.g., functional similarity between regions) and long-range temporal dependencies. ST-MGCN addressed this by encoding multi-dimensional spatial correlations into graphs and using contextual gated RNNs. MGCN-STF builds on this to enhance efficiency.  


## Model Design (Inspired by ST-MGCN)  
- **Multi-Graph Encoding**: Follows ST-MGCN’s core idea of encoding non-Euclidean correlations into three graphs: geographic adjacency, functional similarity, and flow connectivity. Each graph is normalized with self-loops for stable graph convolution.  
- **Temporal Modeling**: Replaces ST-MGCN’s RNN with Transformer Encoders to better capture long-range temporal dependencies via multi-head attention, retaining the goal of "contextual temporal reweighting" but with improved flexibility.  
- **Dual Attention**: Extends ST-MGCN’s contextual gating with two attention mechanisms: time-step attention (reweights temporal features) and branch attention (fuses multi-graph outputs adaptively).  
- **Fusion & Prediction**: Aggregates spatiotemporal features via graph convolution (GConv) for each graph branch, fuses them with branch attention, and maps to predictions via a fully connected layer.  


## Experimental Results  
MGCN-STF was tested on ride-hailing demand forecasting, with RMSE as the metric. Compared to baselines:  
- LSTM (one-way, 3 layers) achieved 29.434.  
- ST-MGCN (no attention) reached 20.3908.  
- MGCN-STF outperformed both with 18.7623, showing 36.2% improvement over LSTM and 8.0% over ST-MGCN (no attention), thanks to Transformer-based temporal modeling and dual attention.  


## Contributions  
- Inherits ST-MGCN’s multi-graph encoding of non-Euclidean correlations.  
- Replaces RNN with Transformer to model long-range temporal dependencies more effectively.  
- Adds dual attention to enhance adaptive feature fusion, addressing ST-MGCN’s limited flexibility.  


## Tools & Environment  
- Hardware: NVIDIA GPU/CPU  
- Software: Python 3.8+, PyTorch 1.18+, PyCharm 2023.2  
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib  


## References  
1. Xu Geng, et al. *Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting*.  
2. Thomas N. Kipf, et al. *Semi-supervised Classification with Graph Convolutional Networks*.  
3. Ashish Vaswani, et al. *Attention Is All You Need*.
