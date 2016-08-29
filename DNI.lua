local DNI, parent = torch.class('nn.DNI', 'nn.Module')

function DNI:__init(src_model, M, M_criterion)
   parent.__init(self)
   self.src_model = src_model
   self.M = M
   self.M_criterion = M_criterion
end

function DNI:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable == 2)
   local input = inputTable[1]
   local label = inputTable[2]

   self.output = self.src_model.forward(input)

   -- Synthetic Gradients
   if label ~= nil then:
      SyntheticGradients = self.M.forward(inputTable)
      self.gradInput = self.src_model:backward(input, SyntheticGradients)
   end
   
   return self.output
end

function DNI:updateGradInput(inputTable, gradOutput)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable == 2)
   local input = inputTable[1]
   local label = inputTable[2]

   -- M learn
   if label ~= nil then:
      local M_grad = self.M_criterion:backward(self.gradInput, gradOutput)
      self.M.backward(inputTable, M_grad)
   end

   return self.gradInput
end