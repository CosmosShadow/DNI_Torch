-- train

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
    model:training()
    parameters, gradParameters = model:getParameters()

    local total_error= 0

    for t = 1, global_iters_each_epochs do
        local inputs, targets, onehot_labels = load_input_target_train()

        if global_use_cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
            onehot_labels = onehot_labels:cuda()
        else
            inputs = inputs:float()
            targets = targets:float()
            onehot_labels = onehot_labels:float()
        end

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            -- forward, backward
            local outputs = model:forward({inputs, onehot_labels})
            local error = criterion:forward(outputs, targets)
            local grad = criterion:backward(outputs, targets)
            model:backward({inputs, onehot_labels}, grad)

            -- normalize
            local batchSize = inputs:size(1)
            gradParameters:div(batchSize)

            total_error= total_error+error
            
            if bPrintInnerError then
                print(error)
            end

            confusion:batchAdd(outputs, targets)

            return error, gradParameters
        end

        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
    end

    if optimMethod == optim.rprop then
        print('==> loss:', total_error/(global_iters_each_epochs*optimState.niter))
    else
        print('==> loss:', total_error/global_iters_each_epochs)
    end

    print(confusion)
    confusion:zero()
    
    return f;
end

