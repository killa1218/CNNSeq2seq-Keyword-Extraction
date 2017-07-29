print("Start time: " .. os.time())

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
local options = {}

local epoch = 10
local batchSize = 128
local lr = 0.01
local lrd = 10
local gpu = true
local kernalWidth = 7
local convLayer = 4
local logInterval = 5000
local fineTune = false
local maxAbsLength = 500
local embDimension = 300
local channelSize = 500

if gpu then
    require 'cunn'
    require 'cutorch'
end

local optimState = {learningRate = lr}

local rawDataset = torch.load('../data/nostem.nopunc.case/discrete/ke20k_training.json.t7')
local validData = torch.load('../data/nostem.nopunc.case/discrete/ke20k_validation.json.t7')
local vocab = torch.load('../data/nostem.nopunc.case/ke20k.nostem.nopunc.case.vocab.t7')

local emb = vocab.idx2vec

if gpu then
    emb = emb:cuda()
end

local data = rawDataset.data
local label = rawDataset.label
local dataSize = #data
local validDataSize = 200 -- TODO 强行减小eval data size


-- Build logger
local logger = optim.Logger('training.cuda.log')
logger:setNames{'training loss', 'validation loss'}
logger:style{'+-', '+-'}


-- Build training data
--local dataset = {}
local batchedDataset = {}
local validDataset = {}

--function dataset:size()
--    return dataSize
--end


function table.slice(tbl, first, last, step)
    local sliced = {}

    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end

    return sliced
end


--for i = 1, dataSize do
--    table.insert(dataset, {data = {emb, data[i]}, label = label[i]})
--end


-- Batch 化数据
-- Training data
print("Making batch training data...")
local i = 1 -- Data index

while i <= dataSize do -- 每次生成一个batch
    local batchData = torch.LongTensor(batchSize * maxAbsLength):fill(1)
    local batchLabel = torch.DoubleTensor(batchSize, maxAbsLength):fill(0)
    --local batchIndex = math.ceil(i / batchSize)

    for dataIndex = 1, batchSize do
        if i <= dataSize then
            local oneData = data[i]
            local oneLabel = label[i]
            local len = oneData:size(1)
            local padSize = maxAbsLength - len
            local startPos = 1 + math.floor(padSize / 2)
            local endPos = maxAbsLength - math.ceil(padSize / 2)
            local dataBase = (dataIndex - 1) * maxAbsLength

            batchData[{{dataBase + startPos, dataBase + endPos}}]:copy(oneData)
            batchLabel[dataIndex][{{startPos, endPos}}]:copy(oneLabel)
            i = i + 1
        else
            i = i + 1
            break
        end

        io.write('\riter: ' .. i)
        io.flush()
    end

    if gpu then
        table.insert(batchedDataset, {data = {emb, batchData:cuda()}, label = batchLabel:cuda()})
    else
        table.insert(batchedDataset, {data = {emb, batchData}, label = batchLabel})
    end
end
print("Finished batch data building.")

-- Validation data
print("Making validation data...")
local validBatchData = torch.LongTensor(validDataSize * maxAbsLength):fill(1)
local validBatchLabel = torch.DoubleTensor(validDataSize, maxAbsLength):fill(0)

i = 1
while i <= validDataSize do -- 每次生成一个batch
    local oneData = validData.data[i]
    local oneLabel = validData.label[i]
    local len = oneData:size(1)
    local padSize = maxAbsLength - len
    local startPos = 1 + math.floor(padSize / 2)
    local endPos = maxAbsLength - math.ceil(padSize / 2)
    local dataBase = (i - 1) * maxAbsLength

    validBatchData[{{dataBase + startPos, dataBase + endPos}}]:copy(oneData)
    validBatchLabel[i][{{startPos, endPos}}]:copy(oneLabel)
    i = i + 1

    io.write('\riter: ' .. i)
    io.flush()
end

local validMask = torch.ne(validBatchData, 1):double():reshape(validDataSize, maxAbsLength)

if gpu then
    validDataset = {data = {emb, validBatchData:cuda() }, label = validBatchLabel:cuda(), mask = validMask:cuda(), num = validMask:sum()}
else
    validDataset = {data = {emb, validBatchData}, label = validBatchLabel, mask = validMask, num = validMask:sum()}
end
print('\rFinished validation data building.')
-- Batch 化数据

------------------------------------------------------------------------------------------------------------------------
-- Build MODEL
-- MODEL Build index
local modelLookup = nn.Sequential()
modelLookup:add(nn.Index(1))


-- MODEL Build reshape
local modelReshape = nn.Reshape(batchSize, maxAbsLength, embDimension)


-- MODEL Build padding
local leftPadSize = math.floor((kernalWidth - 1) / 2)
local rightPadSize = kernalWidth - 1 - leftPadSize
local modelPadding = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))


-- MODEL Build convolution
local modelConvolution = nn.Sequential()
modelConvolution:add(nn.TemporalConvolution(300, channelSize, kernalWidth)):add(nn.LeakyReLU())
for i = 1, convLayer - 3 do
    local localPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
    modelConvolution:add(localPad):add(nn.TemporalConvolution(channelSize, channelSize, kernalWidth)):add(nn.LeakyReLU())
end

local finalPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
modelConvolution:add(finalPad):add(nn.TemporalConvolution(channelSize, math.floor(channelSize / 2), kernalWidth)):add(nn.LeakyReLU())
finalPad = nn.Sequential():add(nn.Padding(2, -leftPadSize)):add(nn.Padding(2, rightPadSize))
modelConvolution:add(finalPad):add(nn.TemporalConvolution(math.floor(channelSize / 2), 1, kernalWidth)):add(nn.Sigmoid())


-- MODEL Build whole model
local model = nn.Sequential()
local padding = nn.Sequential():add(modelLookup):add(modelReshape):add(modelPadding)
model:add(padding):add(modelConvolution):add(nn.Reshape(batchSize, maxAbsLength))


-- Build criterion
local criterion = nn.MSECriterion()
criterion.sizeAverage = false

local eval = nn.AbsCriterion()

if gpu then
    model = model:cuda()
    criterion = criterion:cuda()
    eval = eval:cuda()
end
------------------------------------------------------------------------------------------------------------------------

print(modelConvolution)

-- Trainging
params, gradParams = model:getParameters()
local optimState = {learningRate = lr}
--local errorDataNum = 0

for iter = 1, epoch do
    print('\nTraining epoch ' .. iter .. '\n')

    for i, v in ipairs(batchedDataset) do
        io.write('\rBatch number: ' .. i)
        io.flush()
        local mask = torch.ne(v.data[2], 1):double():reshape(batchSize, maxAbsLength)
        local num = mask:sum() -- Real Abstract length

        if gpu then
            mask = mask:cuda()
        end

        function feval(params)
            gradParams:zero()

            local outputs = model:forward(v.data)
            outputs:cmul(mask)

            assert(outputs:size(1) == v.label:size(1))

            local loss = criterion:forward(outputs, v.label)
            local dloss_doutputs = criterion:backward(outputs, v.label)
            model:backward(v.data, dloss_doutputs)

            return loss, gradParams
        end

        _, l = optim.sgd(feval, params, optimState)

        if i % math.floor(logInterval / batchSize) == 0 or i == 1 then
            -- Log loss and plot
            --local validationOutput = model:forward(validDataset.data)
            --validationOutput:cmul(validDataset.mask)
            --
            --local validationLoss = eval:forward(validationOutput, validDataset.label)
            -- validationLoss = validationLoss / validDataset.num
            local validationLoss = 0

            l[1] = l[1] / num

            logger:add(l, validationLoss)
            logger:plot()
        end
    end
end

print("End time: " .. os.time())