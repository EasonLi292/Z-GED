#!/bin/bash
# Circuit Generator Runner
# Automatically sets the library path for ngspice

export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
python3 "Generated Circuits/circuit_generator.py" "$@"
