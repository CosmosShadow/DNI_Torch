-- input
base_size = 32
image_channel = 1
image_size = 64
noisy_size = 6

-- 分类
classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

-- GPU
global_use_cuda = true
global_GPU_device = 1	-- which one GPU	

bPrintInnerError = false

-- 总的训练数据
global_train_count = 100000000

-- 训练轮次控制
global_iters_each_epochs = 200
global_batch_size = 64

-- 优化方法
optimState = {
	learningRate = 1e-2,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9,
	nesterov = true,
	dampening = 0,
}
optimMethod = optim.sgd