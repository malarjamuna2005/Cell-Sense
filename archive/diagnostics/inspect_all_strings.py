import zlib
import pickletools

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)

strings = []
for opcode, arg, pos in pickletools.genops(decompressed):
    if opcode.name in ('SHORT_BINUNICODE', 'BINUNICODE'):
        strings.append((pos, arg))

print(f"Total strings: {len(strings)}")
for i, (pos, s) in enumerate(strings):
    print(f"{i:2d}: [{pos:6d}] {repr(s)}")
