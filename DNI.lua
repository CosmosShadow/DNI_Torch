local DNI, parent = torch.class('nn.DNI', 'nn.Module')

function DNI:__init(src_model, M)
   parent.__init(self)
   self.src_model = src_model
   self.M = M
end

function DNI:updateOutput(inputTable)
   -- assert(torch.type(inputTable) == 'table')
   -- assert(#inputTable == 2)
   -- local input = inputTable[1]
   -- local label = inputTable[2]
   -- self.output = self.src_model.forward(input)

   self.output = self.src_model:forward(inputTable)
   
   return self.output
end

function DNI:updateGradInput(input, gradOutput)
   self.gradInput = self.src_model:backward(input, gradOutput)
   return self.gradInput
end