import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import random


# -------------------------- 1. 基础配置（随机种子+中文显示） --------------------------
def set_seed(seed=42):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup_chinese_font():
    """设置Matplotlib支持中文显示"""
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


set_seed()
setup_chinese_font()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{DEVICE}")

# -------------------------- 2. 超参数设置 --------------------------
TIME_STEPS = 12  # 时间步长
BATCH_SIZE = 32  # 批次大小
HIDDEN_UNITS = 16  # LSTM隐藏单元数
LEARNING_RATE = 0.01  # 初始学习率
EPOCH = 50  # 训练轮次
NODES = 30  # 节点数量
KEEP_DROP = 0.2  # Dropout保留比例
INPUT_SIZE = 30  # 输入特征数（节点数）
OUTPUT_SIZE = 30  # 输出特征数（节点数）
REGULARIZER = 0.003  # L2正则化系数
TRAIN_EXAMPLES = 6400  # 训练样本数
VAL_EXAMPLES = 800  # 验证样本数
TEST_EXAMPLES = 1600  # 测试样本数
NUM_LAYER = 3  # LSTM层数

# 数据归一化器
scaler = MinMaxScaler(feature_range=(0, 1))


# -------------------------- 3. 数据加载 --------------------------
class TrafficDataset(Dataset):
    """时序数据集类"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(DEVICE)  # [样本数, TIME_STEPS, NODES]
        self.y = torch.FloatTensor(y).to(DEVICE)  # [样本数, NODES]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate(seq, vol):
    """生成时序样本（输入：前TIME_STEPS步，输出：第TIME_STEPS+1步）"""
    X, y = [], []
    max_available = len(seq) - TIME_STEPS  # 避免索引越界
    vol = min(vol, max_available)  # 实际样本数不超过可生成数量

    for i in range(vol):
        X.append(seq[i:i + TIME_STEPS])
        y.append(seq[i + TIME_STEPS])

    # 调整维度
    X = np.array(X, dtype=np.float32).reshape(-1, TIME_STEPS, NODES)
    y = np.array(y, dtype=np.float32).reshape(-1, NODES)
    return X, y


def load_data():
    """加载并划分数据（请根据实际路径修改）"""
    # 数据文件路径（请替换为你的Data.csv实际路径）
    data_path = r'C:\低空经济模型\低空数据\Data.csv'

    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到：{os.path.abspath(data_path)}")

    # 读取并预处理数据
    dataset = pd.read_csv(data_path, header=None, index_col=None)
    values = dataset.values.astype('float32')
    scaled = scaler.fit_transform(values)  # 归一化到[0,1]

    # 划分训练/验证/测试集（7:1:2）
    n_total = len(scaled)
    n_train = int(n_total * 0.7) + 1
    n_val = int(n_total * 0.1)
    n_test = int(n_total * 0.2)

    train = scaled[:n_train, :]
    val = scaled[n_train:n_train + n_val, :]
    test = scaled[n_train + n_val:, :]

    # 生成时序样本
    X_train, y_train = generate(train, TRAIN_EXAMPLES)
    X_val, y_val = generate(val, VAL_EXAMPLES)
    X_test, y_test = generate(test, TEST_EXAMPLES)

    # 封装为DataLoader
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, X_train, y_train, X_val, y_val, X_test, y_test


def load_adj():
    """加载邻接矩阵（请根据实际路径修改）"""
    # 邻接矩阵文件夹路径（请替换为你的实际路径）
    adj_folder = r'C:\低空经济模型\2019-ZJU_SummerResearch-master\DataPreProcessing\邻接矩阵'

    adj_paths = {
        'simi_adj': os.path.join(adj_folder, '30x30_functional_similarity.csv'),
        'dis_adj': os.path.join(adj_folder, '30x30_order_flow.csv'),
        'cont_adj': os.path.join(adj_folder, '30x30_geographic_adjacency.csv')
    }

    # 检查文件是否存在
    for name, path in adj_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"邻接矩阵文件未找到：{os.path.abspath(path)}")

    # 读取并转换为Tensor
    simi_adj = torch.FloatTensor(pd.read_csv(adj_paths['simi_adj'], header=None).values).to(DEVICE)
    dis_adj = torch.FloatTensor(pd.read_csv(adj_paths['dis_adj'], header=None).values).to(DEVICE)
    cont_adj = torch.FloatTensor(pd.read_csv(adj_paths['cont_adj'], header=None).values).to(DEVICE)

    return simi_adj, dis_adj, cont_adj


# -------------------------- 4. 模型核心模块 --------------------------
class GConv(nn.Module):
    """图卷积模块"""

    def __init__(self, input_dim, adj, num_layers=2, nodes=NODES):
        super(GConv, self).__init__()
        self.adj = adj  # [NODES, NODES] 邻接矩阵
        self.nodes = nodes
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()

        # 权重初始化（Xavier初始化）
        for layer in self.conv_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        x: [batch_size, NODES, input_dim]
        返回：[batch_size, NODES, input_dim]
        """
        batch_size = x.shape[0]
        current_x = x

        for layer in self.conv_layers:
            # 邻接矩阵扩展到batch维度 + 矩阵乘法
            adj_expand = self.adj.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, NODES, NODES]
            x_mul_adj = torch.bmm(adj_expand, current_x)  # [batch, NODES, input_dim]
            current_x = self.relu(layer(x_mul_adj))  # 线性变换 + 激活

        return current_x


class Attention(nn.Module):
    """注意力模块（时间步权重重分配）"""

    def __init__(self, time_steps=TIME_STEPS):
        super(Attention, self).__init__()
        self.time_steps = time_steps
        # 注意力权重层
        self.w1 = nn.Linear(time_steps, time_steps)
        self.b1 = nn.Parameter(torch.zeros(time_steps, device=DEVICE))
        self.w2 = nn.Linear(time_steps, time_steps)
        self.b2 = nn.Parameter(torch.zeros(time_steps, device=DEVICE))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 权重初始化
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x, adj):
        """
        x: [batch_size, TIME_STEPS, NODES]
        adj: [NODES, NODES]
        返回：[batch_size, TIME_STEPS, NODES]（注意力加权后）
        """
        batch_size = x.shape[0]

        # 1. 全局池化（节点维度）
        x_pool = torch.sum(x, dim=2)  # [batch_size, TIME_STEPS]

        # 2. GCN处理时序特征
        x_gcn_input = x.permute(0, 2, 1)  # [batch_size, NODES, TIME_STEPS]
        gcn = GConv(input_dim=self.time_steps, adj=adj, num_layers=2, nodes=NODES)
        x_gcn_output = gcn(x_gcn_input)  # [batch_size, NODES, TIME_STEPS]
        x_gcn_pool = torch.sum(x_gcn_output, dim=1)  # [batch_size, TIME_STEPS]

        # 3. 融合池化特征
        x_hat = x_pool + x_gcn_pool  # [batch_size, TIME_STEPS]
        z = x_hat / NODES  # 归一化

        # 4. 计算注意力权重
        tmp_s = self.relu(torch.matmul(z, self.w1.weight) + self.b1)  # [batch_size, TIME_STEPS]
        s = self.sigmoid(torch.matmul(tmp_s, self.w2.weight) + self.b2)  # [batch_size, TIME_STEPS]

        # 5. 重加权输入特征
        s = s.unsqueeze(2)  # [batch_size, TIME_STEPS, 1]
        x_reweight = x * s  # [batch_size, TIME_STEPS, NODES]

        return x_reweight


class LSTMNetwork(nn.Module):
    """LSTM网络（处理单个节点的时序特征）"""

    def __init__(self, input_size=1, hidden_size=HIDDEN_UNITS, num_layers=NUM_LAYER):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else KEEP_DROP
        )

        # 权重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """
        x: [batch_size, TIME_STEPS, 1]
        返回：[batch_size, HIDDEN_UNITS]（最后一个时间步输出）
        """
        out, _ = self.lstm(x)
        return out[:, -1, :]


class STMGCNBranch(nn.Module):
    """STMGCN单分支（注意力+LSTM+GCN）"""

    def __init__(self, adj):
        super(STMGCNBranch, self).__init__()
        self.adj = adj
        self.attention = Attention(time_steps=TIME_STEPS)
        self.lstm = LSTMNetwork()
        self.gcn = GConv(input_dim=HIDDEN_UNITS, adj=adj, num_layers=2, nodes=NODES)

    def forward(self, x):
        """
        x: [batch_size, TIME_STEPS, NODES]
        返回：[batch_size, NODES, HIDDEN_UNITS]
        """
        # 1. 注意力重加权
        x_reweight = self.attention(x, self.adj)

        # 2. 每个节点独立过LSTM
        lstm_outputs = []
        for i in range(NODES):
            node_x = x_reweight[:, :, i:i + 1]  # [batch_size, TIME_STEPS, 1]
            lstm_out = self.lstm(node_x)  # [batch_size, HIDDEN_UNITS]
            lstm_outputs.append(lstm_out.unsqueeze(1))  # [batch_size, 1, HIDDEN_UNITS]

        # 3. 拼接LSTM输出
        lstm_output = torch.cat(lstm_outputs, dim=1)  # [batch_size, NODES, HIDDEN_UNITS]

        # 4. GCN处理
        gcn_output = self.gcn(lstm_output)  # [batch_size, NODES, HIDDEN_UNITS]

        return gcn_output


class STMGCN(nn.Module):
    """完整STMGCN模型（3分支融合）"""

    def __init__(self, simi_adj, dis_adj, cont_adj):
        super(STMGCN, self).__init__()
        # 3个分支（分别对应不同邻接矩阵）
        self.branch_simi = STMGCNBranch(adj=simi_adj)
        self.branch_dis = STMGCNBranch(adj=dis_adj)
        self.branch_cont = STMGCNBranch(adj=cont_adj)

        # 输出层
        self.output_layer = nn.Linear(HIDDEN_UNITS, 1, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        """
        x: [batch_size, TIME_STEPS, NODES]
        返回：[batch_size, NODES]（最终预测）
        """
        # 分支前向传播
        gcn_output1 = self.branch_simi(x)
        gcn_output2 = self.branch_dis(x)
        gcn_output3 = self.branch_cont(x)

        # 融合分支输出
        network_output = gcn_output1 + gcn_output2 + gcn_output3  # [batch_size, NODES, HIDDEN_UNITS]

        # 输出层
        all_output = self.output_layer(network_output)  # [batch_size, NODES, 1]
        all_output = all_output.squeeze(2)  # [batch_size, NODES]

        return all_output


# -------------------------- 5. 训练与评估 --------------------------
def train_model():
    # 加载数据和邻接矩阵
    train_loader, val_loader, test_loader, X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    simi_adj, dis_adj, cont_adj = load_adj()

    # 打印数据信息
    print(f"\n数据加载完成：")
    print(f"训练集：{len(train_loader.dataset)}样本 | {len(train_loader)}批")
    print(f"验证集：{len(val_loader.dataset)}样本 | {len(val_loader)}批")
    print(f"测试集：{len(test_loader.dataset)}样本 | {len(test_loader)}批")
    print(f"输入形状：{X_train.shape} | 输出形状：{y_train.shape}")

    # 初始化模型、损失函数、优化器
    model = STMGCN(simi_adj, dis_adj, cont_adj).to(DEVICE)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=REGULARIZER  # L2正则化
    )

    # 训练监控变量
    loss_cache = []  # 训练损失记录
    val_loss_cache = []  # 验证损失记录
    best_val_rmse = float('inf')  # 最佳验证RMSE
    best_model_path = "best_stmgcn.pth"  # 最佳模型保存路径

    # 开始训练
    print(f"\n开始训练（共{EPOCH}轮）：")
    for epoch in range(EPOCH):
        # -------------------------- 训练阶段 --------------------------
        model.train()
        train_total_loss = 0.0
        for X_batch, y_batch in train_loader:
            # 前向传播
            y_pred = model(X_batch)
            # 计算损失
            loss = criterion(y_pred, y_batch)
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失
            train_total_loss += loss.item() * X_batch.size(0)
            loss_cache.append(loss.item())

        # 平均训练损失
        avg_train_loss = train_total_loss / len(train_loader.dataset)

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        val_total_loss = 0.0
        res_variance_unscaled = []  # 未归一化的预测误差
        with torch.no_grad():  # 禁用梯度计算
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss = criterion(y_pred, y_batch)
                val_total_loss += val_loss.item() * X_batch.size(0)
                val_loss_cache.append(val_loss.item())

                # 反归一化（计算真实尺度误差）
                y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().numpy())
                y_batch_unscaled = scaler.inverse_transform(y_batch.cpu().numpy())
                res_variance_unscaled.append((y_pred_unscaled - y_batch_unscaled) ** 2)

        # 计算验证集指标
        avg_val_loss = val_total_loss / len(val_loader.dataset)
        val_mse_unscaled = np.mean(res_variance_unscaled)
        val_rmse_unscaled = np.sqrt(val_mse_unscaled)

        # 打印训练日志
        print(f"Epoch {epoch + 1:2d}/{EPOCH} | "
              f"训练损失：{avg_train_loss:.6f} | "
              f"验证损失：{avg_val_loss:.6f} | "
              f"验证未归一化MSE：{val_mse_unscaled:.4f} | "
              f"验证未归一化RMSE：{val_rmse_unscaled:.4f}")

        # 保存最佳模型
        if val_rmse_unscaled < best_val_rmse:
            best_val_rmse = val_rmse_unscaled
            torch.save(model.state_dict(), best_model_path)
            print(f"  → 保存最佳模型（当前最佳RMSE：{best_val_rmse:.4f}）")

    # -------------------------- 测试集评估 --------------------------
    print(f"\n===== 测试集评估（基于最佳模型） =====")
    # 加载最佳模型
    best_model = STMGCN(simi_adj, dis_adj, cont_adj).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    # 计算测试集指标
    test_res_variance = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = best_model(X_batch)
            # 反归一化
            y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().numpy())
            y_batch_unscaled = scaler.inverse_transform(y_batch.cpu().numpy())
            test_res_variance.append((y_pred_unscaled - y_batch_unscaled) ** 2)

    test_mse = np.mean(test_res_variance)
    test_rmse = np.sqrt(test_mse)
    print(f"测试集未归一化MSE：{test_mse:.4f}")
    print(f"测试集未归一化RMSE：{test_rmse:.4f}")
    print(f"最佳验证集RMSE：{best_val_rmse:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(loss_cache, label='训练损失', color='blue', alpha=0.7)
    plt.plot(val_loss_cache, label='验证损失', color='red', alpha=0.7)
    plt.title('训练与验证损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 启动训练 --------------------------
if __name__ == "__main__":
    train_model()
