-- list2array.pd_lua
-- Write incoming Pd lists into a named Pd array/table.
--
-- Usage:
--   [list2array myarray]
--
-- Inlet 1:
--   list 0.1 0.2 0.3  -> write full list starting at index 0
--   set foo           -> switch target array to "foo"
--   offset 10         -> next write starts at index 10
--   resize 1          -> resize target array to fit incoming list
--   bang              -> re-sync with target array and post status
--   info              -> output target name and current length

local list2array = pd.Class:new():register("list2array")

function list2array:initialize(sel, atoms)
    atoms = atoms or {}
    self.inlets = 1
    self.outlets = 0
    self.name = type(atoms[1]) == "string" and atoms[1] or nil
    self.offset = 0
    self.do_resize = false
    self.tab = nil
    if self.name then
        self:sync_table()
    end
    return true
end

function list2array:sync_table()
    if not self.name then
        self.tab = nil
        return false
    end
    self.tab = pd.Table:new():sync(self.name)
    if not self.tab then
        self:error("list2array: no Pd array/table named '" .. tostring(self.name) .. "'")
        return false
    end
    return true
end

function list2array:post_status()
    if not self.tab and not self:sync_table() then
        return
    end
    local len = self.tab:length()
    pd.post(string.format("list2array: target=%s length=%d offset=%d resize=%d",
        tostring(self.name), len, self.offset, self.do_resize and 1 or 0))
end

function list2array:write_values(atoms)
    if not self.name then
        self:error("list2array: no target array set")
        return
    end
    if not self.tab and not self:sync_table() then
        return
    end

    local values = {}
    for i = 1, #atoms do
        if type(atoms[i]) ~= "number" then
            self:error("list2array: list input must contain only numbers")
            return
        end
        values[i] = atoms[i]
    end

    local needed = self.offset + #values
    local current_len = self.tab:length()
    if needed > current_len then
        if self.do_resize then
            pd.send(self.name, "resize", {needed})
            if not self:sync_table() then
                return
            end
        else
            self:error(string.format(
                "list2array: incoming list exceeds array length (%d > %d); enable resize with 'resize 1'",
                needed, current_len))
            return
        end
    end

    for i = 1, #values do
        self.tab:set(self.offset + i - 1, values[i])
    end
    self.tab:redraw()
end

function list2array:in_1_list(atoms)
    atoms = atoms or {}
    self:write_values(atoms)
end

function list2array:in_1_set(atoms)
    atoms = atoms or {}
    if type(atoms[1]) ~= "string" then
        self:error("list2array: set needs an array name")
        return
    end
    self.name = atoms[1]
    self:sync_table()
    self:post_status()
end

function list2array:in_1_offset(atoms)
    atoms = atoms or {}
    if type(atoms[1]) ~= "number" then
        self:error("list2array: offset needs a number")
        return
    end
    self.offset = math.max(0, math.floor(atoms[1]))
end

function list2array:in_1_resize(atoms)
    atoms = atoms or {}
    if type(atoms[1]) ~= "number" then
        self:error("list2array: resize needs 0 or 1")
        return
    end
    self.do_resize = atoms[1] ~= 0
end

function list2array:in_1_info()
    self:post_status()
end

function list2array:in_1_bang()
    self:post_status()
end
