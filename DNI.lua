local DNI, parent = torch.class('nn.DNI', 'nn.Module')

function DNI:__init(src_model, M, M_criterion)
   parent.__init(self)
   self.src_model = src_model
   self.M = M
   self.M_criterion = M_criterion
end

function DNI:updateOutput(input)
   self.output = self.src_model:forward(input)

   -- Synthetic Gradients
   if label ~= nil then
      self.SyntheticGradients = self.M:forward(self.output)
      self.gradInput = self.src_model:backward(input, self.SyntheticGradients)
   end

   return self.output
end

function DNI:updateGradInput(input, gradOutput)
   -- M learn
   if label ~= nil then
      local M_grad = self.M_criterion:backward(self.SyntheticGradients, gradOutput)
      self.M:backward(self.output, M_grad)
   end

   return self.gradInput
end