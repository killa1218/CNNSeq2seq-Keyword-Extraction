require 'torch'
local json = require 'json.json'
local gl = require 'glove.glove'

local fileName = ''

local file = io.open(fileName, 'r')



function processK20k(f)
    data = {}

    for line in f.lines() do
        local jo = json.decode(line)
        local abstract = jo['abstract']
        local keywords = jo['keywords']:split(',')
        local s = line:gsub("\t", ""):gsub("^%s+", ""):gsub("%s+$", ""):gsub("%s+", " ")



    end
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
    str = trimAll(str)

end

function tokenize(str)


end
