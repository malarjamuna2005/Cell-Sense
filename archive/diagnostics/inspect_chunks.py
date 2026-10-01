import zlib
import pickletools

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)
print(f"Total decompressed length: {len(decompressed)}")

# Find all PROTO opcodes or chunk boundaries
pos = 0
chunk_idx = 0
while pos < len(decompressed):
    print(f"\n--- Chunk {chunk_idx} at byte offset {pos} ---")
    sub = decompressed[pos:]
    try:
        sub_ops = list(pickletools.genops(sub))
        print(f"Ops count: {len(sub_ops)}")
        for op, arg, p in sub_ops[:15]:
            print(f"  [{p}] {op.name} {repr(arg)}")
        if len(sub_ops) > 15:
            print(f"  ... (and {len(sub_ops) - 15} more ops)")
        # Find where STOP opcode is
        stop_pos = None
        for op, arg, p in sub_ops:
            if op.name == 'STOP':
                stop_pos = p + 1
                break
        if stop_pos is not None:
            pos += stop_pos
        else:
            break
    except Exception as e:
        print(f"Error parsing at pos {pos}: {e}")
        # search next PROTO
        next_proto = decompressed.find(b'\x80', pos + 1)
        if next_proto == -1:
            print(f"Remaining bytes: {len(decompressed) - pos}")
            break
        else:
            pos = next_proto
    chunk_idx += 1
