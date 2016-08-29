-- utils

-- 计算正确个数
function caculateRightCount(input, target, interval, terminal)
    local batch_count = target:size()[1]

    -- 默认为1
    local reward = input[1]:clone():resize(batch_count):fill(1)

    for batch = 1, batch_count do
        for step_target=1, #input/interval do
            step_input = (step_target - 1) * interval + 1
            local _maxValue, _maxIdx = torch.max(input[step_input][batch], 1)
            if _maxIdx[1] ~= target[batch][step_target] then
                reward[batch] = 0
                break
            end
            if target[batch][step_target] == terminal then
                break
            end
        end
    end

    return reward:sum()
end


function index2char(index)
	if index <= 9 then
		return index
	elseif index == 10 then
		return 0
	elseif index < 37 then
		A_index = 65
		return string.char(index-11+A_index)
	else
		print('index error')
	end
end

function prediction2labels(prediction, interval, termial)
	local step_count = #prediction
	local batch_count = prediction[1]:size()[1]

	local output = {}

	for batch_index=1, batch_count do
		label = ''
		for step_index=interval, step_count, interval do
			local _maxValue, _maxIdx = torch.max(prediction[step_index][batch_index], 1)
			if _maxIdx[1] < termial then
				char = index2char(_maxIdx[1])
				label = label..char
			else
				break
			end
		end
		output[batch_index] = label
	end
   
	return output
end

function prediction2labelspossibility(prediction, interval, termial)
	local step_count = #prediction

	label = ''
	possibility = 1.0

	for step_index=interval, step_count, interval do
		local _maxValue, _maxIdx = torch.max(prediction[step_index][1], 1)
		possibility_step = math.exp(_maxValue[1])
		possibility = possibility * possibility_step
		if _maxIdx[1] < termial then
			char = index2char(_maxIdx[1])
			label = label..char
		else
			break
		end
	end

	return label, possibility
end

-- -- 格式化label
function format_label(label)
    if #label > 9 then

        return nil
    end
    local lower_label = string.lower(label)

    -- padding = -1
    local tensor_label = torch.ones(9)*(-1)
    local index = 1

    for i=1, #lower_label do
        local ch = string.sub(lower_label, i, i)

        -- 数字
        if string.is_digit(ch) then
            local value = string.byte(ch) - string.byte('0')
            if value == 0 then
                value = 10
            end
            tensor_label[index] = value
            index = index + 1
        end

        -- 字母
        if string.is_lower_alpha(ch) then
            local value = string.byte(ch) - string.byte('a') + 11
            tensor_label[index] = value
            index = index + 1
        end

        -- 检测label的长度是否超过统一长度
        if index > target_length then
            print('format_label wrong: length is large than '..(target_length-1))
            return nil
        end
    end

    tensor_label[index] = 37

    return tensor_label
end

function format_labels(labels)
	local tensor_labels = torch.ones(#labels, 9)*(-1)
	for i=1, #labels do
		tensor_labels[i] = format_label(labels[i])
	end
	return tensor_labels
end

function targets_masks_2_targets(targets_masks)
	local targets = torch.ones(targets_masks[1][1]:size()[1], target_length) * (-1)
	for step=1,target_length do
		targets:select(2, step):copy(targets[step][1])
	end
	return targets
end

function target2labels(targets, termial)
	local batch_count = targets:size()[1]
	local step_count = targets:size()[2]

	local output = {}

	for batch_index=1, batch_count do
		label = ''
		for step_index=1, step_count do
			value = targets[batch_index][step_index]
			if value < termial then
				char = index2char(value)
				label = label..char
			else
				break
			end
		end
		output[batch_index] = label
	end
   
	return output
end

-- 
function drawBox(img, bbox)
	local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
	local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])
	x1, y1 = math.max(1, x1), math.max(1, y1)
	x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

	local max = img:max()

	for channel=1,3 do
		for i=x1,x2 do
			img[channel][y1][i] = max
			img[channel][y2][i] = max
		end
		for i=y1,y2 do
			img[channel][i][x1] = max
			img[channel][i][x2] = max
		end
	end

	return img
end

function test()
	print(index2char(1))
	print(index2char(10))
	print(index2char(11))
	print(index2char(36))
	print(index2char(37))
end