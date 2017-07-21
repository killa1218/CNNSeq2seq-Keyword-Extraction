-- local th = require('torch')
local json = require('lib.json')
-- local nn = require('nn')

local file = io.open('data/ke20k_validation.json', 'r')

for line in file:lines() do
    local js = json.decode(string.lower(line))
    print(js['title'])
end