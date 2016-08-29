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
dofile 'DNI.lua'
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

print("==> Training")
epoch = 0
while epoch < 1000000000 do
    epoch = epoch + 1
    local time = os.date("%Y_%m_%d_%H_%M_%S", os.time())
    print("\nepoch # " .. epoch..'  '..time..'  ')

    -- 训练
    local error = train()
    optim.adjust_method(error)

    -- 直接保存模型
    if epoch%global_save_parameter_iter == 0 then
        model:float()
        local parameters_path = global_parameter_store_path..'/model_'..time..'.t7'
        torch.save(parameters_path, model)
        print('\nsave to '..parameters_path)
        if global_use_cuda then model:cuda() end
    end

end







