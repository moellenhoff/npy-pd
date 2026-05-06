# npy-pd

Pure Data objects for loading NumPy arrays and writing lists into Pd arrays.

This repository currently includes:

- `npy.pd_lua`: load `.npy` files and access their contents in Pure Data
- `list2array.pd_lua`: write incoming numeric lists into a named Pd array

## Requirements

- Pure Data
- [pdlua](https://agraef.github.io/purr-data/introduction/pd-lua-intro.html)

No other external libraries are required by the Lua objects themselves.

## Installation

1. Put the `.pd_lua` files in a directory that Pure Data can access.
2. Load `pdlua`, for example with `declare -lib pdlua`.
3. Create the objects as `[npy]` and `[list2array]`.

## npy

`[npy]` loads NumPy `.npy` files containing `float32` or `float64` data.

Supported arrays:

- 1D arrays
- 2D arrays
- C-contiguous arrays

### Messages

- `open path/to/file.npy`: load a file
- `bang`: output the full array
- `row N`: output row `N` from a 2D array
- `col N`: output column `N` from a 2D array
- `N`: shorthand for `row N`
- `C R`: output the single value at column `C`, row `R`
- `normalize MIN MAX`: scale future numeric output to the target range
- `meta`: output metadata
- `info`: output only the shape

### Outlets

- left outlet: numeric data
- right outlet: shape and metadata

When a file is loaded, `[npy]` automatically:

- sends metadata to the right outlet
- prints metadata to the Pd console

### Metadata messages

`meta` sends selector-based messages from the right outlet:

- `shape ...`
- `dtype symbol ...`
- `filename symbol ...`
- `rawrange MIN MAX`
- `normalize 0`
- `normalize MIN MAX`

Examples:

- `shape 3 51`
- `dtype symbol <f8`
- `filename symbol /path/to/data.npy`
- `rawrange -0.84 0.93`
- `normalize -1 1`

## list2array

`[list2array myarray]` writes incoming numeric lists into a named Pd array.

This is useful when you want to send data from `[npy]` directly into a Pd array
for plotting, wavetable experiments, envelopes, or control data.

### Messages

- `1 2 3 4`: write the list starting at index `0`
- `offset N`: write starting at index `N`
- `resize 1`: automatically resize the target array if needed
- `set foo`: switch the target array to `foo`
- `info`: print the current target, length, offset, and resize status to the Pd console

`[list2array]` has no outlets. It writes directly to the target array and uses
the Pd console for status and error reporting.

## Notes

- Array indices are zero-based where indexing is exposed by messages.
- `normalize` affects output values, not the stored raw data.
- `dtype <f4` means `float32`; `dtype <f8` means `float64`.
