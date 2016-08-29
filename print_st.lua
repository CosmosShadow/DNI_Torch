---------------------------------------------------------------------------------
print("==> Loading required libraries")
require 'dp'
require 'rnn'
require 'stn'
require 'torch'
require 'xlua'
require 'optim'
require 'image'

require 'Utils'
dofile 'method.lua'
dofile '0_utils.lua'
dofile '0_config.lua'


print("==> Setting: thread, seed")
torch.setnumthreads(1)
torch.manualSeed(123)

-- 初始化method、全局参数
optim.adjust_method()

-- 加载GPU模块
if global_use_cuda then
    require 'cutorch'
    require 'cunn'
end

print("==> Loading scripts and model")
dofile '1_load_data.lua'

dofile '3_loss.lua'
dofile '4_train.lua'

global_trained_parameter_path = 'parameters/tmp.t7'

-- 加载模型
if global_trained_parameter_path and #(global_trained_parameter_path)>0 then
    print('==> load model: '..global_trained_parameter_path)
    model = torch.load(global_trained_parameter_path)
else
    dofile '2_model.lua'
end

-- GPU vs CPU
if global_use_cuda then
    print('==> set model with GPU')
    cutorch.setDevice(global_GPU_device)
    model:cuda()
    criterion:cuda()
else
    print('==> set model with CPU')
    model:float()
    criterion:float()
end

confusion = optim.ConfusionMatrix(classes)

model_st = model:get(1)

local inputs, targets = load_input_target_train()

if global_use_cuda then
    inputs = inputs:cuda()
    targets = targets:cuda()
else
    inputs = inputs:float()
    targets = targets:float()
end

output = model_st:forward(inputs) 

inputs = inputs:float()
output = output:float()

inputs = inputs:add(0.5)
output = output:add(0.5)

for i=1,inputs:size(1) do
    image.save('output/'..i..'.png', inputs[i])
    image.save('output/'..i..'_st.png', output[i])
end








