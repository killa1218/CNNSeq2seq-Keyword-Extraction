require 'torch'
require 'json.json'
glove = require 'glove.glove'

local minAbsLength = 20
local maxAbsLength = 1000
local minWordCount = 10

function string:split(sep)
    local sep, fields = sep or " ", {}
    local pattern = string.format("([^%s]+)", sep)
    
    self:gsub(pattern, function(c) fields[#fields+1] = c end)
    
    return fields
end


function tableSize(tb)
    local count = 0
    
    for k,v in pairs(tb) do
        count = count + 1
    end
    
    return count
end


function calcSimWithOneKeyword(word, phrase) -- Calculate similarity between one word and one phrase
    local emb1 = glove:word2vec(word)
    local phraseParts = trimAll(phrase):split(" ")
    local partNum = tableSize(phraseParts)
    local sum = 0
    
    for _, part in pairs(phraseParts) do -- 将keyphrase中的每个单词与word做内积，然后取内积们的均值
        local emb2 = glove:word2vec(part)
        
        sum = sum + emb1:dot(emb2)
    end
        
    return sum / partNum
end


function calcSimWithPhrases(word, phraseArray) -- Calculate similarity between one word and some phrases
    local len = tableSize(phraseArray)
    local denominator = len * (len + 1) / 2
    local sum = 0
    
    for _, phrase in pairs(phraseArray) do
        local tmpSim = calcSimWithOneKeyword(word, phrase)
        
        sum = sum + tmpSim * len / denominator -- Similarity is weighted average according to position
        len = len - 1
    end
    
    return sum
end


function processKe20k(dir)
    local fileList = {"ke20k_testing.json", "ke20k_training.json", "ke20k_validation.json"}
    -- local data = {} -- {data: [[]]
              --  label: [[]]}
    local word2idx = {}
    local idx2word = {}
    local word2count = {}
    local idx2vec = {torch.FloatTensor(300):zero(), glove:word2vec("UNK")}
    local wordNum = 0 -- Word number in vocab
    local totalAbs = 0
    local validAbs = 0
    local keywordsInAbs = 0
    local totalKeywords = 0
    local longestAbs = 0
    local shortestAbs = 100000
    local absTotalLeng = 0 -- Total length of abs.
    
    local tmpWord2count = {} -- Count words for futher filting by count
    for _, fileName in pairs(fileList) do
        print("")
        print("Processing File: " .. fileName)
        local file = io.open(dir .. "/" .. fileName, 'r')
        
        for l in file:lines() do
            local jo = json.decode(l)
            local abstract = jo['abstract']
            local keywords = jo['keyword']:split(';')
            local found = false
                            
            for i, k in pairs(keywords) do
                if string.find(abstract, k, 1, true) then
                    found = true
                    break
                end
            end
            
            if found then
                local words = tokenize(abstract)
                local wordsNum = tableSize(words)
                
                if wordsNum > minAbsLength and wordsNum < maxAbsLength then
                    for idx, w in pairs(words) do
                        if tmpWord2count[w] == nil then
                            tmpWord2count[w] = 1
                        else
                            tmpWord2count[w] = tmpWord2count[w] + 1
                        end
                    end
                end
            end
        end
        
        file:close()
    end

    for _, fileName in pairs(fileList) do -- Real parse
        print("")
        print("Processing File: " .. fileName)
        local file = io.open(dir .. "/" .. fileName, 'r')
        local dataList = {} -- Used to store data, each data is a Tensor(1 x maxAbsLength) containing word indexs
        local labelList = {} -- Used to store labels, each is a DoubleTensor(1 x maxAbsLength) containing scores

        for l in file:lines() do
            -- l = string.lower(l) -- Convert to lower case
            totalAbs = totalAbs + 1
            
            local jo = json.decode(l)
            local abstract = jo['abstract']
            local keywords = jo['keyword']:split(';')
            local found = false
                            
            for i, k in pairs(keywords) do
                totalKeywords = totalKeywords + 1
                
                if string.find(abstract, k, 1, true) then
                    keywordsInAbs = keywordsInAbs + 1
                    found = true
                end
            end
            
            if found then
                local words = tokenize(abstract)
                local wordsNum = tableSize(words)
                
                if wordsNum > minAbsLength and wordsNum < maxAbsLength then
                    validAbs = validAbs + 1
                    shortestAbs = (shortestAbs > wordsNum and {wordsNum} or {shortestAbs})[1]
                    longestAbs = (longestAbs < wordsNum and {wordsNum} or {longestAbs})[1]
                    absTotalLeng = absTotalLeng + wordsNum
                    
                    local idxArray = {} -- Temp storage for data item
                    local labelArray = {} -- Temp storage for label item
                    
                    for idx, w in pairs(words) do
                        if tmpWord2count[w] > minWordCount then -- Filt words with low word count
                            if word2count[w] == nil then
                                word2count[w] = 1
                                wordNum = wordNum + 1
                                word2idx[w] = wordNum + 2 -- Index 1 is for zero vector, 2 is for nil word
                                idx2word[wordNum + 2] = w
                                local emb = glove:word2vec(w)
                                table.insert(idx2vec, emb)
                                
----                                assert(idx2vec[wordNum + 2]:sum() == glove:word2vec(w):sum())
                                
--                                -- if idx2vec[wordNum + 2]:sum() ~= glove:word2vec(w):sum() then
--                                if w == 'A' then
--                                    local id = word2idx['A']
--                                    print(id)
--                                    print(idx2vec[id])
--                                    print(glove:word2vec('A'))
--                                    print(glove:word2vec('A'):sum())
                                    
----                                    print("")
----                                    print(w)
----                                    print(wordNum)
----                                    print(emb)
----                                    print(idx2vec[3])
----                                    print(word2idx)
----                                    print(word2idx[w])
----                                    print(idx2vec[word2idx[w]])
----                                    print(glove:word2vec(w))
--                                end
                            else
                                word2count[w] = word2count[w] + 1
                            end
                            
                            table.insert(idxArray, word2idx[w])
                            table.insert(labelArray, calcSimWithPhrases(w, keywords))
                        else
                            table.insert(idxArray, 2)
                            table.insert(labelArray, calcSimWithPhrases("UNK", keywords))
                        end
                    end
                    
                    table.insert(dataList, torch.LongTensor(maxAbsLength):fill(1)[{{1, wordsNum}}]:copy(torch.Tensor(idxArray)))
                    table.insert(labelList, torch.DoubleTensor(maxAbsLength):fill(0)[{{1, wordsNum}}]:copy(torch.Tensor(labelArray)))
                end
            end
            
            io.write("\rData processed: " .. totalAbs)
            io.flush()
        end
        
        -- Serialization
        local obj = {data = dataList, label = labelList}
        torch.save(fileName .. ".nostem.nopunc.case.t7", obj)
        torch.save(fileName .. ".nostem.nopunc.case.ascii", obj)
        
        file:close()
    end
    
    local i2v = torch.DoubleTensor(wordNum + 2, 300)
    
    for i, label in pairs(idx2vec) do
        i2v[i] = label
    end
    
    local vocab = {idx2word = idx2word, idx2vec = i2v, word2idx = word2idx, word2count = word2count, size = wordNum}
               -- {idx2word: ["word1", "word2"],
               --  idx2vec: Tensor,
               --  word2idx: {word1: idx,
               --             word2: idx},
               --  word2count: {word1: count,
               --               word2: count}}
               
    torch.save("ke20k.nostem.nopunc.case.vocab.t7", vocab)
    
    print("")
    print("Vocab Size: " .. wordNum)
    print("Total Abstract Number: " .. totalAbs)
    print("Valid Abstract Number: " .. validAbs)
    print("Total Keywords Number: " .. totalKeywords)
    print("Valid Keywords Number: " .. keywordsInAbs)
    print("Longest Abstract: " .. longestAbs)
    print("Shortest Abstract: " .. shortestAbs)
    print("Total data size: " .. absTotalLeng)
end


function trimAll(str)
    return str:gsub("^%s+", ""):gsub("%s+$", ""):gsub("%s+", " ")
end


function processAbbreviation(str)
    local wordList = {"She", "she", "He", "he", "It", "it", "What", "what", "That", "that", "Who", "who", "How", "how", "When", "when", "Where", "where"}
  
    for k, v in pairs(wordList) do
        str = str:gsub("^" .. v .. "'s ", v .. " is "):gsub(" " .. v .. "'s ", " " .. v .. " is ")
    end
    
    str = str:gsub("'ve ", " have "):gsub("'ll ", " will "):gsub("I'm ", "I am "):gsub("'re ", " are ")
    str = str:gsub("'d like ", " would like ")
    
    return str
end


function tokenizeWithPunctuation(str)
    str = processAbbreviation(str)
    str = str:gsub("'t[^a-zA-Z0-9]", "～t～ ")
          :gsub("'d[^a-zA-Z0-9]", "～d～ ")
          :gsub("'s[^a-zA-Z0-9]", "～s～ ")
          :gsub("(%w)-(%w)", "%1～m～%2")
          :gsub("(%w)_(%w)", "%1～u～%2")
    str = str:gsub("(%p)", " %1 ")
    str = trimAll(str)
    str = str:gsub("～t～", "'t"):gsub("～d～", "'d"):gsub("～s～", "'s"):gsub("～m～", "-"):gsub("～u～", "_")

    return str:split(" ")
end


function tokenize(str)
    str = processAbbreviation(str)
    str = str:gsub("'t[^a-zA-Z0-9]", "～t～ ")
          :gsub("'d[^a-zA-Z0-9]", "～d～ ")
          :gsub("'s[^a-zA-Z0-9]", "～s～ ")
          :gsub("(%w)-(%w)", "%1～m～%2")
          :gsub("(%w)_(%w)", "%1～u～%2")
    str = str:gsub("%p", " ")
    str = trimAll(str)
    str = str:gsub("～t～", "'t"):gsub("～d～", "'d"):gsub("～s～", "'s"):gsub("～m～", "-"):gsub("～u～", "_")

    return str:split(" ")
end


local dirName = '/mnt/workspaces/ydtian/CNN-Keyword-Extraction/data'

processKe20k(dirName)
