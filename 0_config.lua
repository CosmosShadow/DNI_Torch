-- config

-- class
classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

-- GPU
global_use_cuda = true
global_GPU_device = 1	-- which one GPU	

-- count, epoch, batch
global_train_count = 100000000
global_iters_each_epochs = 500
global_batch_size = 64

-- optimizing setting
optimState = {
	learningRate = 10,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9,
	nesterov = true,
	dampening = 0,
}
optimMethod = optim.sgd