-- model cnn_location

local function createModel()
    -- 模型
    local model = nn.Sequential()
    model:add(nn.View(32*32):setNumInputDims(3))
    model:add(nn.Linear(32*32, 256))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(256, 64))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(64, 10))
    model:add(nn.LogSoftMax())

    -- 初始化参数
    for k, param in ipairs(model:parameters()) do
        param:uniform(-0.1, 0.1)
    end

    return model
end

model = createModel()










