import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import os
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. Basic Configuration --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"CUDA device enabled: {DEVICE} | Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
else:
    print(f"Using device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- 2. Hyperparameters and Paths --------------------------
TIME_STEPS = 12
BATCH_SIZE = 64
HIDDEN_UNITS = 32  # Must be divisible by nhead (4)
LEARNING_RATE = 0.005
EPOCH = 100
NODES = 30
DROPOUT_RATE = 0.3
REGULARIZER = 0.005
NUM_TRANSFORMER_LAYERS = 2  # Transformer encoder layers
NUM_GCN_LAYERS = 3
DATA_CSV_PATH = 'path/to/your/data/Data.csv'  # Placeholder path
ADJ_DIR = 'path/to/your/adjacency_matrices'  # Placeholder path
scaler = MinMaxScaler(feature_range=(0, 1))


# -------------------------- 3. Data Loading --------------------------
class TrafficDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X).to(DEVICE)
        self.y = torch.FloatTensor(y).to(DEVICE)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.augment:
            shift = np.random.choice([-1, 0, 1])
            if 0 <= idx + shift < len(self.X):
                X = self.X[idx + shift]
            noise = torch.normal(0, 0.01, size=X.shape, device=DEVICE)
            X = X + noise
        return X, y


def generate_seq(seq):
    X, y = [], []
    total_samples = len(seq) - TIME_STEPS
    for i in range(total_samples):
        X.append(seq[i:i + TIME_STEPS])
        y.append(seq[i + TIME_STEPS])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_data():
    dataset = pd.read_csv(DATA_CSV_PATH, header=None, index_col=None)
    values = dataset.values.astype('float32')
    print(f"Raw data dimension: {values.shape}")
    print(
        f"Raw data statistics: Mean={np.mean(values):.6f} | Min={np.min(values):.6f} | Max={np.max(values):.6f}")

    n_train = int(len(values) * 0.7)
    n_val = int(len(values) * 0.1)
    train_raw = values[:n_train]
    val_raw = values[n_train:n_train + n_val]
    test_raw = values[n_train + n_val:]

    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = generate_seq(train_scaled)
    X_val, y_val = generate_seq(val_scaled)
    X_test, y_test = generate_seq(test_scaled)

    train_loader = DataLoader(TrafficDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TrafficDataset(X_val, y_val, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TrafficDataset(X_test, y_test, augment=False),
                             batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data loading completed:")
    print(f"  Training set: {len(train_loader.dataset)} samples | {len(train_loader)} batches")
    print(f"  Validation set: {len(val_loader.dataset)} samples | {len(val_loader)} batches")
    print(f"  Test set: {len(test_loader.dataset)} samples | {len(test_loader)} batches")
    return train_loader, val_loader, test_loader, X_test, y_test


def load_adjacency_matrices():
    adj_files = {
        'functional_similarity': os.path.join(ADJ_DIR, '30x30_functional_similarity.csv'),
        'order_flow': os.path.join(ADJ_DIR, '30x30_order_flow.csv'),
        'geographic_adjacency': os.path.join(ADJ_DIR, '30x30_geographic_adjacency.csv')
    }

    adj_list = []
    for name, path in adj_files.items():
        adj = pd.read_csv(path, header=None, index_col=None).values.astype(np.float32)
        adj_self_loop = adj + np.eye(NODES, dtype=np.float32)
        degree = np.sum(adj_self_loop, axis=1, keepdims=True)
        degree_inv_sqrt = np.power(degree + 1e-8, -0.5)
        adj_norm = degree_inv_sqrt * adj_self_loop * degree_inv_sqrt
        adj_norm = adj_norm / np.linalg.norm(adj_norm, ord=2)
        adj_list.append(torch.FloatTensor(adj_norm).to(DEVICE))
        print(f"✅ {name} adjacency matrix loaded (migrated to {DEVICE})")

    return adj_list


# -------------------------- 4. Model Definition --------------------------
class GConv(nn.Module):
    def __init__(self, input_dim):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(NUM_GCN_LAYERS)])
        self.dropout = nn.Dropout(DROPOUT_RATE).to(DEVICE)
        for layer in self.layers:
            layer.to(DEVICE)
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x, adj):
        current_x = x
        for layer in self.layers:
            x_mul_w = self.dropout(layer(current_x))
            x_gconv = torch.matmul(adj.unsqueeze(0), x_mul_w)
            current_x = torch.relu(x_gconv)
        return current_x


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.w1 = nn.Linear(TIME_STEPS, TIME_STEPS)
        self.w2 = nn.Linear(TIME_STEPS, TIME_STEPS)
        self.attention_weights = None
        self.w1.to(DEVICE)
        self.w2.to(DEVICE)
        nn.init.kaiming_uniform_(self.w1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.w1.bias)
        nn.init.kaiming_uniform_(self.w2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.w2.bias)

    def forward(self, x, adj):
        x_pool = torch.sum(x, dim=2)
        x_gcn_input = x.transpose(1, 2)
        gcn = GConv(input_dim=TIME_STEPS)
        x_gcn_output = gcn(x_gcn_input, adj)
        x_gcn_pool = torch.sum(x_gcn_output, dim=1)
        x_hat = x_pool + x_gcn_pool
        z = x_hat / NODES
        tmp_s = torch.relu(self.w1(z))
        s = torch.sigmoid(self.w2(z))
        self.attention_weights = s.detach().cpu().numpy()
        s = s.unsqueeze(2)
        x_reweight = x * s
        return x_reweight


class TransformerEncoder(nn.Module):
    """Transformer encoder to replace LSTM for temporal feature extraction"""
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # Project 1D input to HIDDEN_UNITS dimension
        self.input_proj = nn.Linear(1, HIDDEN_UNITS).to(DEVICE)
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_UNITS,
            nhead=4,  # Number of attention heads (HIDDEN_UNITS must be divisible by nhead)
            dim_feedforward=HIDDEN_UNITS * 4,
            dropout=DROPOUT_RATE,
            batch_first=True  # Input shape: [batch_size, seq_len, d_model]
        )
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_TRANSFORMER_LAYERS
        ).to(DEVICE)
        self.dropout = nn.Dropout(DROPOUT_RATE).to(DEVICE)

        # Weight initialization
        nn.init.kaiming_uniform_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        """
        x: [batch_size, TIME_STEPS, 1]
        return: [batch_size, HIDDEN_UNITS] (last time step output)
        """
        # Project input to HIDDEN_UNITS dimension
        x_proj = self.dropout(self.input_proj(x))  # [batch_size, TIME_STEPS, HIDDEN_UNITS]
        # Transformer encoder forward pass
        encoder_out = self.transformer_encoder(x_proj)  # [batch_size, TIME_STEPS, HIDDEN_UNITS]
        # Return last time step output (consistent with LSTM's output format)
        return self.dropout(encoder_out[:, -1, :])


class MGCN_STFBranch(nn.Module):
    def __init__(self):
        super(MGCN_STFBranch, self).__init__()
        self.attention = Attention()
        # Replace LSTM list with Transformer list
        self.transformer_list = nn.ModuleList([TransformerEncoder() for _ in range(NODES)])
        self.gcn = GConv(input_dim=HIDDEN_UNITS)
        self.gcn_features = None

    def forward(self, x, adj):
        x_reweight = self.attention(x, adj)
        transformer_outputs = []
        for transformer in self.transformer_list:
            node_idx = len(transformer_outputs)
            node_feat = x_reweight[:, :, node_idx:node_idx + 1]  # [batch_size, TIME_STEPS, 1]
            transformer_out = transformer(node_feat)  # [batch_size, HIDDEN_UNITS]
            transformer_outputs.append(transformer_out.unsqueeze(1))  # [batch_size, 1, HIDDEN_UNITS]
        transformer_concat = torch.cat(transformer_outputs, dim=1)  # [batch_size, NODES, HIDDEN_UNITS]
        gcn_out = self.gcn(transformer_concat, adj)
        self.gcn_features = gcn_out.detach().cpu().numpy()
        return gcn_out


class MGCN_STF(nn.Module):
    def __init__(self):
        super(MGCN_STF, self).__init__()
        self.branch_simi = MGCN_STFBranch()
        self.branch_dis = MGCN_STFBranch()
        self.branch_cont = MGCN_STFBranch()
        self.branch_attn = nn.Sequential(
            nn.Linear(HIDDEN_UNITS * 3, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_UNITS, 3),
            nn.Softmax(dim=2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_UNITS // 2, 1)
        )
        self.branch_attn.to(DEVICE)
        self.output_layer.to(DEVICE)
        for m in self.branch_attn.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x, adj_list):
        out_simi = self.branch_simi(x, adj_list[0])
        out_dis = self.branch_dis(x, adj_list[1])
        out_cont = self.branch_cont(x, adj_list[2])
        cat_outs = torch.cat([out_simi, out_dis, out_cont], dim=2)
        branch_weights = self.branch_attn(cat_outs).unsqueeze(3)
        out_stack = torch.stack([out_simi, out_dis, out_cont], dim=2)
        fusion_out = torch.sum(out_stack * branch_weights, dim=2)
        output = self.output_layer(fusion_out).squeeze(2)
        return output


# -------------------------- 5. Evaluation Metrics --------------------------
def calculate_metrics(y_true, y_pred, phase="test"):
    """Calculate evaluation metrics with improved MAPE handling"""
    # Dimension check
    assert y_true.ndim == 2 and y_pred.ndim == 2, f"Inputs must be 2D tensors, current y_true dim: {y_true.ndim}"
    assert y_true.shape[1] == NODES and y_pred.shape[1] == NODES, \
        f"Node count mismatch, expected {NODES}, current y_true nodes: {y_true.shape[1]}"

    # Denormalization
    y_true_unscaled = scaler.inverse_transform(y_true.cpu().numpy())
    y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().numpy())

    # Print true value statistics for test phase
    if phase == "test":
        print(f"\n[{phase} Set True Value Statistics]")
        print(
            f"  Mean: {np.mean(y_true_unscaled):.6f} | Min: {np.min(y_true_unscaled):.6f} | Max: {np.max(y_true_unscaled):.6f}")
        print(
            f"  1st percentile: {np.percentile(y_true_unscaled, 1):.6f} | 10th percentile: {np.percentile(y_true_unscaled, 10):.6f}")

        # Print sample comparisons
        print(f"\n[First 5 Sample Comparisons (Partial Nodes)]")
        sample_idx = min(5, len(y_true_unscaled))
        node_idx = min(3, NODES)
        for i in range(sample_idx):
            true_str = " | ".join([f"Node {j}: {y_true_unscaled[i, j]:.6f}" for j in range(node_idx)])
            pred_str = " | ".join([f"Node {j}: {y_pred_unscaled[i, j]:.6f}" for j in range(node_idx)])
            print(f"Sample {i + 1} | True: {true_str} | Pred: {pred_str}")

    # Calculate basic metrics
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    # Improved MAPE calculation
    truncate_val = max(np.percentile(y_true_unscaled, 1), 1e-6)
    y_true_clipped = np.maximum(y_true_unscaled, truncate_val)
    error_ratio = np.abs((y_true_unscaled - y_pred_unscaled) / y_true_clipped)
    error_ratio_clipped = np.clip(error_ratio, 0, 10)  # Limit max error ratio to 10 (1000%)
    mape = np.mean(error_ratio_clipped) * 100  # Convert to percentage

    # Pearson correlation coefficient
    pearson_list = []
    for node_idx in range(NODES):
        true_node = y_true_unscaled[:, node_idx]
        pred_node = y_pred_unscaled[:, node_idx]
        if np.var(true_node) > 1e-8 and np.var(pred_node) > 1e-8:
            corr, _ = pearsonr(true_node, pred_node)
            pearson_list.append(corr)
    pearson_corr = np.mean(pearson_list) if pearson_list else 0.0
    pearson_corr = np.nan_to_num(pearson_corr, nan=0.0)

    # Node-level metrics
    node_rmse = []
    node_mae = []
    for node_idx in range(NODES):
        true_node = y_true_unscaled[:, node_idx]
        pred_node = y_pred_unscaled[:, node_idx]
        node_rmse.append(np.sqrt(mean_squared_error(true_node, pred_node)))
        node_mae.append(mean_absolute_error(true_node, pred_node))
    node_rmse = np.array(node_rmse)
    node_mae = np.array(node_mae)

    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Pearson_Corr': pearson_corr,
        'Node_RMSE': node_rmse, 'Node_MAE': node_mae
    }


# -------------------------- 6. Visualization --------------------------
class Visualizer:
    def __init__(self, save_dir='mgcn_stf_visualization'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Pearson_Corr': []}
        self.val_metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Pearson_Corr': []}

    def update_metrics(self, train_metric, val_metric):
        for key in self.train_metrics.keys():
            self.train_metrics[key].append(train_metric[key])
            self.val_metrics[key].append(val_metric[key])

    def plot_multi_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Pearson_Corr']
        titles = ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
                  'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE, %)',
                  'Pearson Correlation Coefficient']
        for i, (key, title) in enumerate(zip(metrics, titles)):
            axes[i].plot(self.train_metrics[key], label='Training Set', linewidth=2, color='#1f77b4')
            axes[i].plot(self.val_metrics[key], label='Validation Set', linewidth=2, color='#ff7f0e', linestyle='--')
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Epoch', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=10)
        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'multi_metrics_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Multi-metric curves saved to {self.save_dir}")

    def plot_node_error_heatmap(self, node_rmse, node_mae):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        rmse_mat = node_rmse.reshape(5, 6)
        mae_mat = node_mae.reshape(5, 6)
        im1 = ax1.imshow(rmse_mat, cmap='Reds', aspect='auto', vmin=0, vmax=np.max(node_rmse) * 0.8)
        ax1.set_title('Node RMSE Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='RMSE')
        im2 = ax2.imshow(mae_mat, cmap='Blues', aspect='auto', vmin=0, vmax=np.max(node_mae) * 0.8)
        ax2.set_title('Node MAE Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='MAE')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'node_error_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Node error heatmaps saved to {self.save_dir}")

    def plot_attention_weights(self, attention_weights, epoch):
        if attention_weights is None or len(attention_weights) == 0:
            return
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i in range(min(5, len(attention_weights))):
            axes[i].bar(range(TIME_STEPS), attention_weights[i], alpha=0.7, color='#2ca02c')
            axes[i].set_title(f'Sample {i + 1} Attention Weights (Epoch {epoch})', fontsize=10)
            axes[i].set_xlabel('Time Step', fontsize=9)
            axes[i].set_ylim(0, 1)
        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'attention_weights_epoch{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gcn_tsne(self, gcn_features, epoch):
        if gcn_features is None or len(gcn_features) == 0:
            return
        feat = gcn_features[0].reshape(NODES, -1)
        feat_2d = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(feat)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=range(NODES), cmap='viridis', s=100)
        plt.colorbar(scatter, label='Node Index')
        plt.title(f'GCN Features t-SNE Visualization (Epoch {epoch})', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f'gcn_tsne_epoch{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ GCN t-SNE plot saved (Epoch {epoch})")


# -------------------------- 7. Training Process --------------------------
def train_model():
    train_loader, val_loader, test_loader, X_test, y_test = load_data()
    adj_list = load_adjacency_matrices()

    model = MGCN_STF()
    for name, param in model.named_parameters():
        if param.device != DEVICE:
            print(f"⚠️ Parameter {name} device mismatch, forcing migration to {DEVICE}")
            param.data = param.data.to(DEVICE)

    criterion = nn.MSELoss(reduction='none').to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=REGULARIZER,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )

    visualizer = Visualizer()
    best_val_rmse = float('inf')
    best_model_path = os.path.join(visualizer.save_dir, 'best_mgcn_stf.pth')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n===== Training Configuration =====")
    print(f"Device: {DEVICE} | Total epochs: {EPOCH} | Batch size: {BATCH_SIZE}")
    print(f"Model parameters: {total_params / 1e4:.1f}K | Initial learning rate: {LEARNING_RATE}")
    print(f"Transformer layers: {NUM_TRANSFORMER_LAYERS} | Hidden units: {HIDDEN_UNITS} | Attention heads: 4")
    if DEVICE.type == 'cuda':
        print(f"Current memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB (CUDA only)")
    print(f"==================================\n")

    for epoch in range(EPOCH):
        model.train()
        train_loss_total = 0.0
        train_y_true = []
        train_y_pred = []

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch, adj_list)
            loss = criterion(y_pred, y_batch)
            error_weight = 1 + (loss / (loss.max() + 1e-8))
            weighted_loss = (loss * error_weight).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_y_true.append(y_batch.detach())
            train_y_pred.append(y_pred.detach())
            train_loss_total += weighted_loss.item() * X_batch.size(0)

        train_y_true = torch.cat(train_y_true, dim=0)
        train_y_pred = torch.cat(train_y_pred, dim=0)
        train_metric = calculate_metrics(train_y_true, train_y_pred, phase="train")

        model.eval()
        val_loss_total = 0.0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch, adj_list)
                loss = criterion(y_pred, y_batch).mean()
                val_loss_total += loss.item() * X_batch.size(0)
                val_y_true.append(y_batch.detach())
                val_y_pred.append(y_pred.detach())

        val_y_true = torch.cat(val_y_true, dim=0)
        val_y_pred = torch.cat(val_y_pred, dim=0)
        val_metric = calculate_metrics(val_y_true, val_y_pred, phase="val")

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metric['RMSE'])
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"⚠️  Learning rate adjusted: {prev_lr:.8f} → {current_lr:.8f}")

        visualizer.update_metrics(train_metric, val_metric)
        if (epoch + 1) % 10 == 0:
            visualizer.plot_multi_metrics()
            visualizer.plot_attention_weights(model.branch_simi.attention.attention_weights, epoch + 1)
            visualizer.plot_gcn_tsne(model.branch_simi.gcn_features, epoch + 1)

        avg_train_loss = train_loss_total / len(train_loader.dataset)
        if val_metric['RMSE'] < best_val_rmse:
            best_val_rmse = val_metric['RMSE']
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch + 1:3d} | ✅ Best model updated | Train loss: {avg_train_loss:.6f} | Val RMSE: {best_val_rmse:.4f} | Val MAPE: {val_metric['MAPE']:.2f}%")
        else:
            print(
                f"Epoch {epoch + 1:3d} | Train loss: {avg_train_loss:.6f} | Val RMSE: {val_metric['RMSE']:.4f} | Val MAPE: {val_metric['MAPE']:.2f}%")

    # Test phase evaluation
    print(f"\n===== Test Set Evaluation =====")
    best_model = MGCN_STF()
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(DEVICE)
    best_model.eval()

    test_y_true = []
    test_y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = best_model(X_batch, adj_list)
            test_y_true.append(y_batch.detach())
            test_y_pred.append(y_pred.detach())

    test_y_true = torch.cat(test_y_true, dim=0)
    test_y_pred = torch.cat(test_y_pred, dim=0)
    test_metric = calculate_metrics(test_y_true, test_y_pred, phase="test")

    visualizer.plot_node_error_heatmap(test_metric['Node_RMSE'], test_metric['Node_MAE'])
    print(f"\nTest set final metrics:")
    print(f"  RMSE: {test_metric['RMSE']:.4f} | MAE: {test_metric['MAE']:.4f}")
    print(f"  MAPE: {test_metric['MAPE']:.2f}% | Pearson Correlation: {test_metric['Pearson_Corr']:.4f}")
    print(f"  Best validation RMSE: {best_val_rmse:.4f}")
    print(f"  All results saved to: {os.path.abspath(visualizer.save_dir)}")


# -------------------------- 8. Start Training --------------------------
if __name__ == "__main__":
    required_files = [
        DATA_CSV_PATH,
        os.path.join(ADJ_DIR, '30x30_functional_similarity.csv'),
        os.path.join(ADJ_DIR, '30x30_order_flow.csv'),
        os.path.join(ADJ_DIR, '30x30_geographic_adjacency.csv')
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Critical file missing: {os.path.abspath(file)}, please check path correctness")

    if DEVICE.type == 'cuda':
        print("✅ CUDA device enabled, training will be accelerated with CUDA")
    else:
        print(f"⚠️ CUDA not available, training will use CPU")

    train_model()