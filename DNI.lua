local DNI, parent = torch.class('nn.DNI', 'nn.Module')

function DNI:__init(src_model, M, M_criterion, M_lr_scale)
   parent.__init(self)
   self.src_model = src_model
   self.M = M
   self.M_criterion = M_criterion or nn.MSECriterion()
   self.M_lr_scale = M_lr_scale or 1e4
end

function DNI:updateOutput(input)
   self.output = self.src_model:forward(input)

   -- Synthetic Gradients
   self.SyntheticGradients = self.M:forward(self.output)
   self.gradInput = self.src_model:backward(input, self.SyntheticGradients)

   return self.output
end

function DNI:updateGradInput(input, gradOutput)
   -- M learn
   local M_error = self.M_criterion:forward(self.SyntheticGradients, gradOutput) / self.SyntheticGradients:nElement()
   local M_grad = self.M_criterion:backward(self.SyntheticGradients, gradOutput)
   self.M:backward(self.output, M_grad*self.M_lr_scale)

   return self.gradInput
end