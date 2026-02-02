# Proprietary Software
# Copyright (c) 2026 Nigel Phillips
# All rights reserved.
# Unauthorized copying, modification, distribution, or use is prohibited.

import wfdb
import matplotlib
import matplotlib.pyplot as plt

print("matplotlib backend:", matplotlib.get_backend())
print("Loading record...")

record = wfdb.rdrecord("data/mitdb/100")
signal = record.p_signal[:, 0]

print("Signal length:", len(signal))
print("Plotting...")

plt.figure(figsize=(12, 4))
plt.plot(signal[:3000])
plt.title("MIT-BIH Record 100 (first 3000 samples)")
plt.xlabel("Sample")
plt.ylabel("mV")
plt.tight_layout()

out = "ecg_plot.png"
plt.savefig(out, dpi=150)
print("Saved:", out)

# Try to show as well
plt.show()
print("Done.")
