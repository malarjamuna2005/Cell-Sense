import zlib

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

data = zlib.decompress(raw)
print(f"Total size: {len(data)}")

# Let's inspect around byte 1170
print("Bytes 1160 to 1250:")
print(data[1160:1250])
print("\nFormatted:")
for i in range(1160, min(len(data), 1300), 16):
    chunk = data[i:i+16]
    hex_str = " ".join(f"{b:02x}" for b in chunk)
    ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"{i:6d}: {hex_str:<48}  {ascii_str}")
