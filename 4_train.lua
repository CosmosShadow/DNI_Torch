-- train

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
    model:training()
    parameters, gradParameters = model:getParameters()

    local total_error= 0

    for t = 1, global_iters_each_epochs do
        local inputs, targets = load_input_target_train()

        if global_use_cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        else
            inputs = inputs:float()
            targets = targets:float()
        end

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            -- forward, backward
            local outputs = model:forward(inputs)
            local error = criterion:forward(outputs, targets)
            local grad = criterion:backward(outputs, targets)
            model:backward(inputs, grad)

            -- print(outputs[1])
            -- print(grad[1])
            -- print(error)

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

