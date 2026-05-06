-- npy.pd_lua
-- Pure Data external to read NumPy .npy files (1D or 2D, float32/float64)
--
-- Usage:  [npy filename.npy]   or send message "open filename.npy"
--
-- Outlets:
--   1 (left)  : data values as floats, sent as a list per row
--               (for 1D arrays: one list with all values)
--               (for 2D arrays: one list per row, bang between rows)
--   2 (right) : info list: shape as "rows cols" (or "cols" for 1D)
--               sent before data
--
-- Example messages to inlet 1:
--   bang             -> output the entire array
--   open foo.npy     -> load new file
--   row 3            -> output only row 3 (0-indexed)
--   col 12           -> output only column 12 (0-indexed)
--   3                -> shorthand for row 3 (0-indexed)
--   12 2             -> output single value at col 12, row 2 (0-indexed)
--   normalize -1 1   -> map output values to a target range
--   meta             -> output filename, dtype, shape, ranges

local npy = pd.Class:new():register("npy")

-- -------------------------------------------------------------------------
-- helpers
-- -------------------------------------------------------------------------

local function read_u16_le(s, pos)
    local lo = string.byte(s, pos)
    local hi = string.byte(s, pos + 1)
    return lo + hi * 256
end

local function read_u32_le(s, pos)
    local b0 = string.byte(s, pos)
    local b1 = string.byte(s, pos + 1)
    local b2 = string.byte(s, pos + 2)
    local b3 = string.byte(s, pos + 3)
    return b0 + b1*256 + b2*65536 + b3*16777216
end

-- Parse a float32 from 4 bytes (IEEE 754 little-endian)
local function bytes_to_f32(b1, b2, b3, b4)
    local n = b1 + b2*256 + b3*65536 + b4*16777216
    local sign = (n >= 0x80000000) and -1 or 1
    local exp  = ((n >> 23) & 0xFF) - 127
    local mant = n & 0x7FFFFF
    if exp == -127 then
        return sign * math.ldexp(mant, -149)  -- denormal
    elseif exp == 128 then
        return (mant == 0) and sign * math.huge or 0/0  -- inf/nan
    else
        return sign * math.ldexp(mant + 0x800000, exp - 23)
    end
end

-- Parse a float64 from 8 bytes (IEEE 754 little-endian)
-- Lua numbers are already doubles, so we just use string.unpack
local function bytes_to_f64(data, pos)
    -- string.unpack available in Lua 5.3+
    local val = string.unpack("<d", data, pos)
    return val
end

-- -------------------------------------------------------------------------
-- npy parser
-- -------------------------------------------------------------------------

local function parse_npy(filename)
    local f, err = io.open(filename, "rb")
    if not f then
        return nil, "cannot open file: " .. (err or filename)
    end

    -- Read and check magic
    local magic = f:read(6)
    if not magic or magic ~= "\x93NUMPY" then
        f:close()
        return nil, "not a valid .npy file (bad magic)"
    end

    -- Version
    local ver = f:read(2)
    local major = string.byte(ver, 1)
    -- local minor = string.byte(ver, 2)  -- not used

    -- Header length
    local hlen_bytes = f:read(2)
    local hlen
    if major == 1 then
        hlen = read_u16_le(hlen_bytes, 1)
    else
        -- v2.0: 4-byte header length
        local hlen_extra = f:read(2)
        hlen = read_u32_le(hlen_bytes .. hlen_extra, 1)
    end

    -- Read header string
    local header = f:read(hlen)
    if not header then
        f:close()
        return nil, "truncated header"
    end

    -- Parse dtype  (we support <f4, <f8, >f4, >f8, float32, float64)
    local dtype = header:match("'descr'%s*:%s*'([^']+)'")
            or header:match('"descr"%s*:%s*"([^"]+)"')
    if not dtype then
        f:close()
        return nil, "cannot parse dtype from header"
    end

    -- Parse shape
    local shape_str = header:match("'shape'%s*:%s*%(([^%)]+)%)")
               or header:match('"shape"%s*:%s*%(([^%)]+)%)')
    if not shape_str then
        f:close()
        return nil, "cannot parse shape from header"
    end

    local dims = {}
    for d in shape_str:gmatch("%d+") do
        dims[#dims + 1] = tonumber(d)
    end
    if #dims == 0 then
        f:close()
        return nil, "0-dimensional arrays not supported"
    end

    -- fortran order?
    local fortran = header:match("'fortran_order'%s*:%s*True") ~= nil
    if fortran then
        f:close()
        return nil, "Fortran-order arrays not supported (use np.ascontiguousarray)"
    end

    -- Total number of elements
    local n_elem = 1
    for _, d in ipairs(dims) do n_elem = n_elem * d end

    -- Read raw data
    local itemsize
    local is_f64 = false
    if dtype:match("f8") or dtype:match("float64") then
        itemsize = 8
        is_f64   = true
    elseif dtype:match("f4") or dtype:match("float32") then
        itemsize = 4
    else
        f:close()
        return nil, "unsupported dtype: " .. dtype .. " (need float32 or float64)"
    end

    local raw = f:read(n_elem * itemsize)
    f:close()
    if not raw or #raw < n_elem * itemsize then
        return nil, "truncated data section"
    end

    -- Decode values into a flat Lua table
    local values = {}
    if is_f64 then
        for i = 1, n_elem do
            local pos = (i - 1) * 8 + 1
            values[i] = bytes_to_f64(raw, pos)
        end
    else
        for i = 1, n_elem do
            local pos = (i - 1) * 4 + 1
            local b1, b2, b3, b4 = string.byte(raw, pos, pos + 3)
            values[i] = bytes_to_f32(b1, b2, b3, b4)
        end
    end

    return { dims = dims, values = values, dtype = dtype }
end

-- -------------------------------------------------------------------------
-- reshape helper: get row i (0-indexed) from a 2D flat array
-- -------------------------------------------------------------------------
local function get_row(values, ncols, row_idx)
    local out = {}
    local offset = row_idx * ncols
    for c = 1, ncols do
        out[c] = values[offset + c]
    end
    return out
end

local function get_col(values, nrows, ncols, col_idx)
    local out = {}
    for r = 0, nrows - 1 do
        out[r + 1] = values[r * ncols + col_idx + 1]
    end
    return out
end

local function minmax(values)
    local vmin = values[1]
    local vmax = values[1]
    for i = 2, #values do
        local v = values[i]
        if v < vmin then vmin = v end
        if v > vmax then vmax = v end
    end
    return vmin, vmax
end

-- -------------------------------------------------------------------------
-- Pd-Lua object
-- -------------------------------------------------------------------------

function npy:initialize(sel, atoms)
    atoms = atoms or {}
    self.inlets  = 1
    self.outlets = 2   -- outlet 1: data, outlet 2: info
    self.data    = nil -- parsed npy table
    self.filename = nil
    self.raw_min = nil
    self.raw_max = nil
    self.norm_out_min = nil
    self.norm_out_max = nil

    -- optional creation argument: filename
    if type(atoms[1]) == "string" then
        self:load(atoms[1])
    end
    return true
end

function npy:load(filename)
    local result, err = parse_npy(filename)
    if not result then
        self:error("npy: " .. err)
        return false
    end
    self.data     = result
    self.filename = filename
    self.raw_min, self.raw_max = minmax(result.values)
    local dims    = result.dims
    if #dims == 1 then
        pd.post(string.format("npy: loaded %s  shape=(%d,)  dtype=%s",
                              filename, dims[1], result.dtype))
    else
        pd.post(string.format("npy: loaded %s  shape=(%d,%d)  dtype=%s",
                              filename, dims[1], dims[2], result.dtype))
    end
    self:send_meta()
    self:post_meta()
    return true
end

function npy:has_normalization()
    return self.norm_out_min ~= nil and self.norm_out_max ~= nil
end

function npy:scale_value(v)
    if not self:has_normalization() then
        return v
    end
    local in_min = self.raw_min
    local in_max = self.raw_max
    local out_min = self.norm_out_min
    local out_max = self.norm_out_max

    if in_min == nil or in_max == nil then
        return v
    end
    if math.abs(in_max - in_min) < 1e-12 then
        return (out_min + out_max) * 0.5
    end
    local t = (v - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)
end

function npy:scale_list(values)
    if not self:has_normalization() then
        return values
    end
    local out = {}
    for i = 1, #values do
        out[i] = self:scale_value(values[i])
    end
    return out
end

function npy:send_info()
    if not self.data then return end
    local dims = self.data.dims
    if #dims == 1 then
        self:outlet(2, "list", {dims[1]})
    else
        self:outlet(2, "list", {dims[1], dims[2]})
    end
end

function npy:send_all()
    if not self.data then
        self:error("npy: no file loaded")
        return
    end
    local dims   = self.data.dims
    local values = self.data.values

    self:send_info()

    if #dims == 1 then
        -- Send all values as a single list
        self:outlet(1, "list", self:scale_list(values))
    else
        -- Send each row as a list
        local nrows = dims[1]
        local ncols = dims[2]
        for r = 0, nrows - 1 do
            local row = get_row(values, ncols, r)
            self:outlet(1, "list", self:scale_list(row))
        end
    end
end

function npy:send_row_index(row_index)
    if not self.data then
        self:error("npy: no file loaded")
        return
    end

    local dims = self.data.dims
    local r = type(row_index) == "number" and math.floor(row_index) or 0
    if #dims == 1 then
        local n = dims[1]
        if r < 0 or r >= n then
            self:error(string.format("npy: index %d out of range (0..%d)", r, n - 1))
            return
        end
        self:outlet(1, "float", {self:scale_value(self.data.values[r + 1])})
    else
        local nrows = dims[1]
        local ncols = dims[2]
        if r < 0 or r >= nrows then
            self:error(string.format("npy: row %d out of range (0..%d)", r, nrows - 1))
            return
        end
        self:outlet(1, "list", self:scale_list(get_row(self.data.values, ncols, r)))
    end
end

function npy:send_value_at(col_index, row_index)
    if not self.data then
        self:error("npy: no file loaded")
        return
    end

    local dims = self.data.dims
    local c = type(col_index) == "number" and math.floor(col_index) or nil
    local r = type(row_index) == "number" and math.floor(row_index) or nil
    if c == nil or r == nil then
        self:error("npy: need list 'col row'")
        return
    end

    if #dims == 1 then
        self:error("npy: col row access requires a 2D array")
        return
    end

    local nrows = dims[1]
    local ncols = dims[2]
    if c < 0 or c >= ncols then
        self:error(string.format("npy: col %d out of range (0..%d)", c, ncols - 1))
        return
    end
    if r < 0 or r >= nrows then
        self:error(string.format("npy: row %d out of range (0..%d)", r, nrows - 1))
        return
    end

    local index = r * ncols + c + 1
    self:outlet(1, "float", {self:scale_value(self.data.values[index])})
end

function npy:send_col_index(col_index)
    if not self.data then
        self:error("npy: no file loaded")
        return
    end

    local dims = self.data.dims
    local c = type(col_index) == "number" and math.floor(col_index) or 0
    if #dims == 1 then
        self:error("npy: col access requires a 2D array")
        return
    end

    local nrows = dims[1]
    local ncols = dims[2]
    if c < 0 or c >= ncols then
        self:error(string.format("npy: col %d out of range (0..%d)", c, ncols - 1))
        return
    end

    self:outlet(1, "list", self:scale_list(get_col(self.data.values, nrows, ncols, c)))
end

function npy:send_meta()
    if not self.data then
        self:error("npy: no file loaded")
        return
    end

    local dims = self.data.dims
    if #dims == 1 then
        self:outlet(2, "shape", {dims[1]})
    else
        self:outlet(2, "shape", {dims[1], dims[2]})
    end
    self:outlet(2, "dtype", {"symbol", self.data.dtype})
    if self.filename then
        self:outlet(2, "filename", {"symbol", self.filename})
    end
    self:outlet(2, "rawrange", {self.raw_min, self.raw_max})
    if self:has_normalization() then
        self:outlet(2, "normalize", {self.norm_out_min, self.norm_out_max})
    else
        self:outlet(2, "normalize", {0})
    end
end

function npy:post_meta()
    if not self.data then
        self:error("npy: no file loaded")
        return
    end

    local dims = self.data.dims
    if #dims == 1 then
        pd.post(string.format("npy: shape %d", dims[1]))
    else
        pd.post(string.format("npy: shape %d %d", dims[1], dims[2]))
    end
    pd.post(string.format("npy: dtype %s", tostring(self.data.dtype)))
    if self.filename then
        pd.post(string.format("npy: filename %s", tostring(self.filename)))
    end
    pd.post(string.format("npy: rawrange %.12g %.12g", self.raw_min, self.raw_max))
    if self:has_normalization() then
        pd.post(string.format("npy: normalize %.12g %.12g", self.norm_out_min, self.norm_out_max))
    else
        pd.post("npy: normalize off")
    end
end

-- bang: (re)send all data
function npy:in_1_bang()
    self:send_all()
end

-- open <filename>: load a new file
function npy:in_1_open(atoms)
    atoms = atoms or {}
    if type(atoms[1]) ~= "string" then
        self:error("npy: open needs a filename argument")
        return
    end
    self:load(atoms[1])
end

-- row <n>: output single row n (0-indexed) to outlet 1
function npy:in_1_row(atoms)
    atoms = atoms or {}
    self:send_row_index(atoms[1])
end

-- col <n>: output single column n (0-indexed) to outlet 1
function npy:in_1_col(atoms)
    atoms = atoms or {}
    self:send_col_index(atoms[1])
end

-- two-number list: output single value at col,row
function npy:in_1_list(atoms)
    atoms = atoms or {}
    if #atoms == 2 and type(atoms[1]) == "number" and type(atoms[2]) == "number" then
        self:send_value_at(atoms[1], atoms[2])
        return
    end
    self:error("npy: list input expects exactly two numbers: 'col row'")
end

-- bare float: output single row/index directly
function npy:in_1_float(f)
    self:send_row_index(f)
end

-- normalize <min> <max>: scale future output values to the target range
function npy:in_1_normalize(atoms)
    atoms = atoms or {}
    if not self.data then
        self:error("npy: no file loaded")
        return
    end
    if type(atoms[1]) ~= "number" or type(atoms[2]) ~= "number" then
        self:error("npy: normalize needs two numbers, e.g. 'normalize -1 1'")
        return
    end
    self.norm_out_min = atoms[1]
    self.norm_out_max = atoms[2]
end

-- info: send shape to outlet 2 without sending data
function npy:in_1_info()
    self:send_info()
end

-- meta: send filename, dtype, shape, raw range, normalization mode
function npy:in_1_meta()
    self:send_meta()
    self:post_meta()
end
