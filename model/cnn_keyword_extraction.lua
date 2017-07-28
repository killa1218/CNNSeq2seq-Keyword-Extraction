print("Start time： " .. os.time())

-- require('mobdebug').start()
require 'nn'
require 'torch'
require 'optim'

-- vocab structure:
--  {
--      idx2word: ["word1", "word2"],
--      idx2vec: Tensor,
--      word2idx: {
--          word1: idx,
--          word2: idx
--      },
--      word2count: {
--          word1: count,
--          word2: count
--      }
--  }

-- Options
local epoch = 10
local batchSize = 1
local lr = 0.01
local lrd = 10
local gpu = false
local kernalWidth = 7
local convLayer = 4
local logInterval = 100
local fineTune = false

local optimState = {learningRate = lr}

local rawDataset = torch.load('../data/nostem.nopunc.case/discrete/ke20k_training.t7')
local evalData = torch.load('../data/nostem.nopunc.case/discrete/ke20k_validation.t7')
local vocab = torch.load('../data/nostem.nopunc.case/ke20k.nostem.nopunc.case.vocab.t7')

local emb = vocab.idx2vec

local data = rawDataset.data
local label = rawDataset.label
local dataSize = #data

local sample = data[2]

-- Build logger
local logger = optim.Logger('training.log')
logger:setNames{'training loss', 'validation loss'}
logger:style{'+-', '+-'}

-- Build training data
local dataset = {}
local batchedDataset = {}

function dataset:size()
    return dataSize
end

function table.slice(tbl, first, last, step)
    local sliced = {}

    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end

    return sliced
end

for i = 1, dataSize do
    table.insert(dataset, {data = {emb, data[i]}, label = label[i]})
end

-- Batch 化数据
local i = 1
local tmpDataTable = {}
local tmpLabelTable = {}

while batchSize ~= 1 and i < dataSize do
    table.insert(tmpDataTable, emb:index(1, data[i]))
    table.insert(tmpLabelTable, label[i])

    if i % batchSize == 0 or i == dataSize then
        table.insert(batchedDataset, {data = torch.cat(tmpDataTable, 1), label = torch.cat(tmpLabelTable, 2)})
        tmpDataTable = {}
        tmpLabelTable = {}
    end

    i = i + 1
end

--local evaluationData = torch.cat()

-- Batch 化数据

-- Build model
-- Build index
local lookup = nn.Sequential()
lookup:add(nn.Index(1))

-- Build padding
local leftPadSize = math.floor((kernalWidth - 1) / 2)
local rightPadSize = kernalWidth - 1 - leftPadSize
local pad = nn.Sequential():add(nn.Padding(1, -leftPadSize)):add(nn.Padding(1, rightPadSize))

-- Build convolution
local cnn = nn.Sequential()
--cnn:add(nn.TemporalRowConvolution(300, kernalWidth, 30)):add(nn.ReLU()):add(nn.Linear(300, 1024)):add(nn.Linear(1024, 10)):add(nn.SoftMax())
cnn:add(nn.TemporalConvolution(300, 500, kernalWidth)):add(nn.LeakyReLU())
for i = 1, convLayer - 3 do
    local lpad = nn.Sequential():add(nn.Padding(1, -leftPadSize)):add(nn.Padding(1, rightPadSize))
    cnn:add(lpad):add(nn.TemporalConvolution(500, 500, kernalWidth)):add(nn.LeakyReLU())
end
local lpad = nn.Sequential():add(nn.Padding(1, -leftPadSize)):add(nn.Padding(1, rightPadSize))
cnn:add(lpad):add(nn.TemporalConvolution(500, 250, kernalWidth)):add(nn.LeakyReLU())
lpad = nn.Sequential():add(nn.Padding(1, -leftPadSize)):add(nn.Padding(1, rightPadSize))
cnn:add(lpad):add(nn.TemporalConvolution(250, 1, kernalWidth)):add(nn.Sigmoid())

-- Build whole model
local model = nn.Sequential()
local padding = nn.Sequential():add(lookup):add(pad)
model:add(padding):add(cnn):add(nn.Reshape())


-- Build criterion
--local criterion = nn.MSECriterion()
local criterion = nn.AbsCriterion()
criterion.sizeAverage = false
local eval = nn.AbsCriterion()


print(cnn)
--local sampleData = dataset[300][1]
--local sampleLabel = dataset[300][2]
--print(sampleLabel:size())
--local output = model:forward(sampleData)
----print("Output: ", output)
----print("Label: ", sampleLabel)
--local loss = criterion:forward(output, sampleLabel)
--print(loss)


-- Trainer
--local trainer = nn.StochasticGradient(model, criterion)
--trainer.learningRate = 0.01


-- Trainging
--for i = 1, epoch do
--    print("Training epoch " .. i)
--    trainer:train(dataset)
--end
params, gradParams = model:getParameters()
local optimState = {learningRate = lr}
local errorDataNum = 0

for iter = 1, epoch do
    for i, v in ipairs(dataset) do
        --local mask = torch.ne(v.data, 0):typeAs(torch.DoubleTensor())

        function feval(params)
            gradParams:zero()

            local outputs = model:forward(v.data)
            --outputs:cmul(mask)

            if outputs:size(1) == v.label:size(1) then
                local loss = criterion:forward(outputs, v.label)
                local dloss_doutputs = criterion:backward(outputs, v.label)
                model:backward(v.data, dloss_doutputs)

                return loss, gradParams
            else
                errorDataNum = errorDataNum + 1
                return 0, gradParams
            end
        end

        _, l = optim.sgd(feval, params, optimState)

        if i % logInterval == 0 or i == 1 then
            -- Log loss and plot
            --local output = model:forward(v.data)
            --local loss = criterion:forward(output, v.label)
            -- local validationLoss = -- Calculate validation loss
            local validationLoss = 0
            print("Error Data Number: " .. errorDataNum)

            logger:add(l, validationLoss)
            logger:plot()
        end
    end
end

print("End time： " .. os.time())