using Flux

# 准备训练数据
# 输入是两个随机数，输出是它们的和
function generate_data(num_samples)
    x = rand(-10:10, 2, num_samples)
    y = sum(x, dims=1)
    return x, y
end

# 定义模型
model = Chain(
    Dense(2, 10, σ),  # 2个输入特征，10个隐藏单元
    Dense(10, 1)      # 10个隐藏单元，1个输出
)

# 定义损失函数
loss(x, y) = Flux.mse(model(x), y)

# 准备训练数据
num_samples = 1000
x, y = generate_data(num_samples)

# 设置优化器
opt = Descent(0.1)  # 学习率为 0.1

# 训练模型
for i in 1:10000
    Flux.train!(loss, Flux.params(model), [(x, y)], opt)
    if i % 1000 == 0
        println("Epoch $i, Loss: $(loss(x, y))")
    end
end

# 测试模型
x_test, y_test = generate_data(10)
predictions = model(x_test)
println("Predictions: ", predictions)
println("True Values: ", y_test)
