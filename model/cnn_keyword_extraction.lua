require 'nn'
require 'torch'
require 'optim'

local epoch = 10
local bachSize = 50
local lr = 0.01
local lrd = 10
local gpu = false
local kernalWidth = 7

local optimState = {learningRate = lr}

local data
local label

local lookup = nn.Sequential()
lookup:add(nn.Index(1))

local cnn = nn.Sequential()
cnn:add(nn.TemporalRowConvolutional())


local batchInputs = torch.DoubleTensor(batchSize, inputs) -- or CudaTensor for GPU training
local batchLabels = torch.DoubleTensor(batchSize)         -- or CudaTensor for GPU training

for i = 1, batchSize do
   local input = torch.randn(2)     -- normally distributed example in 2d
   local label
   if input[1] * input[2] > 0 then  -- calculate label for XOR function
      label = -1
   else
      label = 1
   end
   batchInputs[i]:copy(input)
   batchLabels[i] = label
end

params, gradParams = model:getParameters()

for iter = 1, epoch do
   -- local function we give to optim
   -- it takes current weights as input, and outputs the loss
   -- and the gradient of the loss with respect to the weights
   -- gradParams is calculated implicitly by calling 'backward',
   -- because the model's weight and bias gradient tensors
   -- are simply views onto gradParams
   function feval(params)
      gradParams:zero()

      local outputs = model:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      model:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
   end
   optim.sgd(feval, params, optimState)
end