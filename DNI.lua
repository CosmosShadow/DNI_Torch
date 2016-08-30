local DNI, parent = torch.class('nn.DNI', 'nn.Module')

function DNI:__init(src_model, M, M_criterion)
   parent.__init(self)
   self.src_model = src_model
   self.M = M
   self.M_criterion = M_criterion or nn.MSECriterion()
   self.current_M_confidence = 0
end

function DNI:updateOutput(input)
   self.output = self.src_model:forward(input)

   -- Synthetic Gradients
   self.SyntheticGradients = self.M:forward(self.output)
   self.gradInput = self.src_model:backward(input, self.SyntheticGradients*self.current_M_confidence)

   return self.output
end

function DNI:updateGradInput(input, gradOutput)
   -- M learn
   local M_error = self.M_criterion:forward(self.SyntheticGradients, gradOutput) / self.SyntheticGradients:nElement()
   local M_grad = self.M_criterion:backward(self.SyntheticGradients, gradOutput)
   self.M:backward(self.output, M_grad)

   M_error = M_error*1e7
   -- print(M_error)

   if M_error < 1 then
      self.current_M_confidence = (1 - M_error)*0.001
   else
      self.current_M_confidence = 0
   end

   -- print(self.current_M_confidence)

   -- self.gradInput = self.src_model:backward(input, gradOutput)

   return self.gradInput
end