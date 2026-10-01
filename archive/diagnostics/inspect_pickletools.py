import zlib
import pickletools
import io

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)
print(f"Decompressed: {len(decompressed)} bytes", flush=True)

globals_found = set()
strings_found = []

for opcode, arg, pos in pickletools.genops(decompressed):
    if opcode.name in ('GLOBAL', 'STACK_GLOBAL', 'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8'):
        if opcode.name == 'GLOBAL':
            globals_found.add(arg)
        elif opcode.name in ('SHORT_BINUNICODE', 'BINUNICODE'):
            strings_found.append((pos, arg))

print("\nGLOBALs found:")
for g in sorted(globals_found):
    print(f"  {g}")

# Let's inspect strings
print(f"\nTotal strings parsed: {len(strings_found)}")
print("\nFirst 40 strings:")
for pos, s in strings_found[:40]:
    print(f"  [{pos}] {repr(s)}")

# Let's find strings related to features or models
print("\nInteresting strings (containing features, sklearn, estimators, soh, volt, etc.):")
for pos, s in strings_found:
    s_lower = str(s).lower()
    if any(term in s_lower for term in ['feature', 'volt', 'curr', 'temp', 'cycle', 'soh', 'regressor', 'scaler', 'pipeline', 'gradient', 'forest', 'tree', 'extra', 'xgb', 'lgb', 'step']):
        print(f"  [{pos}] {repr(s)}")
