-- model

local function createModel()
    local M1 = nn.Sequential()
    M1:add(nn.Linear(256, 1024))
    M1:add(nn.BatchNormalization(1024))
    M1:add(nn.ReLU())
    M1:add(nn.Linear(1024, 1024))
    M1:add(nn.BatchNormalization(1024))
    M1:add(nn.ReLU())
    M1:add(nn.Linear(1024, 256))

    local M2 = M1:clone()

    local M3 = nn.Sequential()
    M3:add(nn.Linear(10, 256))
    M3:add(nn.BatchNormalization(256))
    M3:add(nn.ReLU())
    M3:add(nn.Linear(256, 256))
    M3:add(nn.BatchNormalization(256))
    M3:add(nn.ReLU())
    M3:add(nn.Linear(256, 10))

    local model = nn.Sequential()

    -- full DNI
    -- model:add(nn.DNI(nn.Sequential():add(nn.Linear(32*32, 256)):add(nn.ReLU()), M1, nn.MSECriterion(), 1e4))
    -- model:add(nn.DNI(nn.Sequential():add(nn.Linear(256, 256)):add(nn.ReLU()), M2, nn.MSECriterion(), 1e4))
    -- model:add(nn.DNI(nn.Linear(256, 10), M3, nn.MSECriterion(), 1e4))

    -- one DNI
    model:add(nn.Linear(32*32, 256))
    model:add(nn.ReLU())
    model:add(nn.Linear(256, 64))
    model:add(nn.ReLU())
    model:add(nn.DNI(nn.Linear(64, 10), M3, nn.MSECriterion(), 1e3))

    -- init parameters
    for k, param in ipairs(model:parameters()) do
        param:uniform(-0.1, 0.1)
    end

    return model
end

model = createModel()










