import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pymysql
from torch.optim.lr_scheduler import StepLR

# 数据库连接
def get_data_from_db():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='123456',
        database='steamdb',
        charset='utf8mb4'
    )
    query = "SELECT * FROM app01_gamerating"
    df = pd.read_sql(query, connection)
    connection.close()
    return df


# 数据预处理：将user_id和game_id转换为编码
def preprocess_data(df):
    user_encoder = LabelEncoder()
    game_encoder = LabelEncoder()

    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['game_id_encoded'] = game_encoder.fit_transform(df['game_id'])

    # 将评分进行类别化（例如：评分 [1, 2, 3, 4, 5] 转化为类别 [0, 1, 2, 3, 4]）
    df['rating_category'] = df['rating'] - 1  # 将评分 1-5 转化为 0-4

    return df, user_encoder, game_encoder


# 定义神经网络模型
class RecommenderNN(nn.Module):
    def __init__(self, num_users, num_games, num_classes=5, embedding_dim=50, dropout_rate=0.2):
        super(RecommenderNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.game_embedding = nn.Embedding(num_games, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.fc = nn.Linear(embedding_dim * 2, num_classes)  # 输出层大小为类别数

    def forward(self, user, game):
        user_emb = self.user_embedding(user)
        game_emb = self.game_embedding(game)
        x = torch.cat([user_emb, game_emb], dim=-1)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)
        return x  # 返回 logits，未经 softmax 处理


# 训练模型
def train_model(df, num_users, num_games, num_classes=5, embedding_dim=50, dropout_rate=0.2, epochs=5, batch_size=32, lr=0.001):
    model = RecommenderNN(num_users, num_games, num_classes, embedding_dim, dropout_rate)
    criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调整：每训练一定轮次后，降低学习率
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # 每3个epoch，学习率减少为原来的0.1倍

    # 转换数据
    users = torch.tensor(df['user_id_encoded'].values, dtype=torch.long)
    games = torch.tensor(df['game_id_encoded'].values, dtype=torch.long)
    ratings = torch.tensor(df['rating_category'].values, dtype=torch.long)  # 使用类别标签

    # 训练循环
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(users.size(0))
        for i in range(0, users.size(0), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            user_batch, game_batch, rating_batch = users[indices], games[indices], ratings[indices]

            # 前向传播
            output = model(user_batch, game_batch)
            loss = criterion(output, rating_batch)  # CrossEntropyLoss直接使用模型的输出和目标标签

            # 反向传播
            loss.backward()
            optimizer.step()

        scheduler.step()  # 调整学习率
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()}")

    return model


# 生成推荐
def recommend(model, user_id, user_encoder, game_encoder, top_k=10):
    model.eval()

    # 获取用户的编码
    user_encoded = user_encoder.transform([user_id])[0]

    # 对所有游戏进行评分预测
    all_game_ids = np.arange(len(game_encoder.classes_))
    game_ids_encoded = torch.tensor(all_game_ids, dtype=torch.long)
    user_batch = torch.tensor([user_encoded] * len(all_game_ids), dtype=torch.long)

    with torch.no_grad():
        logits = model(user_batch, game_ids_encoded)  # 获取模型的logits输出
        probabilities = torch.softmax(logits, dim=-1)  # 将 logits 转化为概率
        predicted_classes = torch.argmax(probabilities, dim=-1).numpy()  # 获取最大概率对应的类别

    # 获取推荐的游戏ID，按照类别排序
    top_k_game_indices = predicted_classes.argsort()[-top_k:][::-1]
    top_k_game_ids = game_encoder.inverse_transform(top_k_game_indices)

    return top_k_game_ids


# 主程序
def rec(user_id):
    # 获取数据
    df = get_data_from_db()

    # 数据预处理
    df, user_encoder, game_encoder = preprocess_data(df)

    # 定义神经网络
    num_users = len(user_encoder.classes_)
    num_games = len(game_encoder.classes_)

    # 训练模型
    model = train_model(df, num_users, num_games)

    # 推荐
    recommended_games = recommend(model, user_id, user_encoder, game_encoder)

    return recommended_games

'''
# 计算用户之间的相似度（使用余弦相似度）
def compute_user_similarity(df, num_users):
    # 创建用户-游戏评分矩阵
    user_game_matrix = np.zeros((num_users, len(df['game_id_encoded'].unique())))
    for row in df.itertuples():
        user_game_matrix[row.user_id_encoded, row.game_id_encoded] = row.rating_category

    # 计算用户之间的余弦相似度
    user_similarity = cosine_similarity(user_game_matrix)
    return user_similarity
'''

