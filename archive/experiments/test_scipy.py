import sys

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy error:", e)

try:
    import scipy
    print("scipy:", scipy.__version__)
    import scipy.stats
    print("scipy.stats imported!")
except Exception as e:
    print("scipy error:", e)
