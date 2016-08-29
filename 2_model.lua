-- model

local function createModel()
    local M1 = nn.Sequential()
    M1:add(nn.JoinTable(2))
    M1:add(nn.Linear(256+10, 256))
    M1:add(nn.ReLU())
    M1:add(nn.Linear(256, 256))

    local M2 = nn.Sequential()
    M2:add(nn.JoinTable(2))
    M2:add(nn.Linear(64+10, 64))
    M2:add(nn.ReLU())
    M2:add(nn.Linear(64, 64))

    local M3 = nn.Sequential()
    M3:add(nn.JoinTable(2))
    M3:add(nn.Linear(10+10, 10))
    M3:add(nn.ReLU())
    M3:add(nn.Linear(10, 10))

    -- 模型
    local model = nn.Sequential()
    model:add(nn.DNI(nn.Sequential():add(nn.Linear(32*32, 256)):add(nn.ReLU()), M1))
    model:add(nn.DNI(nn.Sequential():add(nn.Linear(256, 64)):add(nn.ReLU()), M2))
    model:add(nn.DNI(nn.Sequential():add(nn.Linear(64, 10)):add(nn.LogSoftMax()), M3))

    -- 初始化参数
    for k, param in ipairs(model:parameters()) do
        param:uniform(-0.1, 0.1)
    end

    return model
end

model = createModel()










