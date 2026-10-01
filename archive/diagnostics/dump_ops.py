import zlib
import pickletools

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)

ops = list(pickletools.genops(decompressed))
print(f"Total ops: {len(ops)}")
for i, (opcode, arg, pos) in enumerate(ops):
    print(f"{i:3d}: [{pos:7d}] {opcode.name:<25} {repr(arg)}")
