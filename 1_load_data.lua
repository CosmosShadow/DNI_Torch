-- load data

---------------------------------------------------------------------------------
print("==> Loading train data")

data_dir = 'mnist'
data_train_path = data_dir..'/train_32x32.t7'
data_test_path = data_dir..'/test_32x32.t7'

local data_trian = torch.load(data_train_path, 'ascii')
local data_test = torch.load(data_test_path, 'ascii')

function load_input_target_train()

    local inputs = torch.zeros(global_batch_size, 32*32)
    local labels = torch.zeros(global_batch_size)
    local onehot_labels = torch.zeros(global_batch_size, 10)

    for i=1,global_batch_size do
        local tsize = data_trian.data:size(1)
        local random_index = math.random(math.min(tsize, global_train_count))

        local img = data_trian.data[random_index]
        inputs[i]:copy(img)

        labels[i] = data_trian.labels[random_index]
        onehot_labels[i][labels[i]] = 1
    end

    -- 归一化处理
    inputs = inputs:div(255.0):add(-0.5)

    return inputs, labels, onehot_labels
end





