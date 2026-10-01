import zlib
import pickle
import io

with open('cellsense_final_soh_model.joblib', 'rb') as f:
    raw = f.read()

decompressed = zlib.decompress(raw)

class TracingUnpickler(pickle._Unpickler):
    def load(self):
        self.mark = 0
        self.stack = []
        self.memo = {}
        read = self.readline
        readinto = self.readinto
        while True:
            key = self.read(1)
            if not key:
                break
            # print opcode
            op = ord(key)
            if op == ord('\x93'): # STACK_GLOBAL
                print("About to execute STACK_GLOBAL. Stack top 4:", [repr(x)[:60] for x in self.stack[-4:]])
            self.dispatch[op](self)
            if key == b'.':
                break
        return self.stack.pop()

try:
    unp = TracingUnpickler(io.BytesIO(decompressed))
    obj = unp.load()
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
