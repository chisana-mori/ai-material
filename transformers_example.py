import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================================================================================
# 1. 环境设置：模拟一个微型GPT模型
# 为了清晰地展示计算过程，我们不直接加载2B参数的模型，而是构建一个结构相同但尺寸极小的版本。
# 这能让我们轻松查看每一步的张量（向量）变化。
# ======================================================================================

# 配置与 `向量的深度之旅` 章节中的示例保持一致
d_model = 4  # 向量维度
vocab_size = 10  # 假设我们的词汇表很小
sequence_length = 2  # 输入序列长度 "an apple"

# 定义一个简化的、单层的Transformer解码器块
class TinyDecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 注意力机制的权重矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 前馈网络 (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 残差连接 1
        residual = x
        x = self.ln1(x)

        # 步骤 2: QKV 线性变换
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 步骤 3: 计算注意力分数
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(d_model)

        # 创建一个上三角掩码，防止看到未来的token
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, -float('inf'))

        # 步骤 4: Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 步骤 5: 生成上下文向量
        context_vector = attn_weights @ v

        # 应用残差连接
        x = residual + context_vector

        # 残差连接 2
        residual = x
        x = self.ln2(x)

        # 步骤 6: FFN 非线性变换
        ffn_output = self.ffn(x)

        # 应用残差连接
        x = residual + ffn_output

        return x, q, k, v, attn_scores, attn_weights, context_vector

# 模拟一个完整的微型GPT
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_block = TinyDecoderBlock(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # 步骤 1: 初始向量 (Embeddings)
        token_embeddings = self.embedding(idx)
        # (在真实模型中，这里还会加上位置编码)

        # 通过解码器层处理
        final_vectors, q, k, v, scores, weights, context = self.decoder_block(token_embeddings)

        # 步骤 7 (部分): 生成预测
        logits = self.output_head(final_vectors)

        return logits, token_embeddings, q, k, v, scores, weights, context

# 为了结果可复现，设置随机种子
torch.manual_seed(42)

# ======================================================================================
# 2. 训练演示：严格参照 `向量的深度之旅` 章节顺序
# ======================================================================================
print("="*50)
print("🚀 开始向量的深度之旅：一次完整的训练迭代")
print("="*50)

# 实例化模型
model = TinyGPT(vocab_size, d_model)

# 准备输入数据
# 假设 "an" -> 0, "apple" -> 1, "a" -> 2
input_tokens = torch.tensor([[0, 1]])  # 输入: "an apple"
target_token_for_apple = torch.tensor([2]) # 当输入是 "an apple" 时，我们希望模型在 "apple" 的位置上预测出 "a"

# --- 前向传播 (Forward Pass) ---
logits, embeddings, q, k, v, scores, weights, context = model(input_tokens)

print("
[第一步：初始向量 (Embeddings)]")
print("模型接收到 'an apple' (token ID: [0, 1])，并查找它们的初始向量。")
print(f"初始向量 (Embeddings):
{embeddings.detach()}
")

print("
[第二步：QKV线性变换]")
print("每个输入向量分别与 W_q, W_k, W_v 矩阵相乘，生成 Query, Key, Value 向量。")
print(f"Query (查询) 向量:
{q.detach()}
")
print(f"Key (键) 向量:
{k.detach()}
")
print(f"Value (值) 向量:
{v.detach()}
")

print("
[第三步：计算注意力分数]")
print("用每个位置的 Query 向量去和它能看到的所有位置的 Key 向量做点积，并进行缩放。")
print("由于有掩码，'apple' (位置1) 只能关注 'an' (位置0) 和它自身。")
print(f"原始注意力分数 (应用掩码后):
{scores.detach()}
")

print("
[第四步：Softmax归一化]")
print("将分数转换为0到1之间、总和为1的权重，代表注意力分布。")
print(f"注意力权重 (Attention Weights):
{weights.detach()}
")
print("解读: 在处理 'apple' 这个词时，模型将 ~14% 的注意力放在 'an' 上，~86% 的注意力放在 'apple' 自身。")

print("
[第五步：生成上下文向量]")
print("用注意力权重对所有 Value 向量进行加权求和，得到融合了上下文的新向量。")
print(f"上下文向量 (Context Vector):
{context.detach()}
")

print("
[第六步：FFN, 残差连接 & 层归一化]")
print("向量经过前馈网络、残差连接和层归一化，进行深度加工，得到最终输出向量。")
final_vector_for_apple = logits[:, 1, :] # 我们只关心在 "apple" 位置的输出
print(f"在 'apple' 位置的最终输出向量:
{final_vector_for_apple.detach()}
")

print("
[第七步：损失函数计算]")
print("模型使用最终向量预测词汇表中每个词的概率，并与真实目标 'a' (token ID: 2) 对比。")
# 我们只关心在 "apple" 位置的预测，因为这是我们有答案的地方
logits_for_apple = logits[:, 1, :]
loss = F.cross_entropy(logits_for_apple, target_token_for_apple)
print(f"模型预测的Logits (在 'apple' 位置):
{logits_for_apple.detach()}
")
print(f"真实目标 Token ID: {target_token_for_apple.item()}")
print(f"计算出的交叉熵损失 (Cross-Entropy Loss): {loss.item():.4f}
")
print("解读: 损失值是一个衡量 '预测错误程度' 的标量。我们的目标是通过调整权重来让它变小。")

# --- 反向传播 (Backward Pass) ---

# 清空旧的梯度
model.zero_grad()

print("
[第八步：反向传播与梯度计算]")
print("损失值开始反向传播，使用链式法则计算出每个权重相对于损失的梯度。")
loss.backward()
print("梯度计算完成。梯度指明了每个权重应该调整的方向和幅度。
")

print("
[第九步：权重更新与优化]")
print("优化器 (如Adam) 使用计算出的梯度来更新模型的每一个权重。")
# 这是一个简化的手动更新过程，用以演示
learning_rate = 0.01
print(f"以 '注意力层' 的 W_q 权重为例 (学习率: {learning_rate}):")
wq_weight_before = model.decoder_block.W_q.weight.data.clone()
print(f"更新前的 W_q 权重 (部分):
{wq_weight_before[0, :].numpy()}...")

# 手动执行一步梯度下降
with torch.no_grad():
    for param in model.parameters():
        if param.grad is not None:
            param -= learning_rate * param.grad

wq_weight_after = model.decoder_block.W_q.weight.data.clone()
print(f"
更新后的 W_q 权重 (部分):
{wq_weight_after[0, :].numpy()}...")
print("可以看到权重已经被微调。这个过程会在数百万个样本上重复数十亿次，模型的能力从而得到提升。")

print("
" + "="*50)
print("🎉 演示完成！")
print("="*50)
