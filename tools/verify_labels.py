import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

# Default dataset path; adjust if you want to point to another pickle
DATASET_PATH = "rlc_dataset/filter_dataset.pkl"
NUM_SAMPLES = 3


def plot_pole_zero_map(ax, poles, zeros, title):
    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", s=100, color="red", label="Poles")
    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", s=100, facecolors="none", edgecolors="blue", label="Zeros")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Real Axis (σ)")
    ax.set_ylabel("Imag Axis (jω)")
    ax.set_title(title)
    ax.legend()


def plot_bode(ax_mag, ax_phase, freqs, H_spice, poles, zeros, gain, title):
    ax_mag.semilogx(freqs, 20 * np.log10(np.abs(H_spice)), label="SPICE Sim", color="black", alpha=0.5, linewidth=2)
    ax_phase.semilogx(freqs, np.degrees(np.angle(H_spice)), label="SPICE Sim", color="black", alpha=0.5, linewidth=2)

    w = 2 * np.pi * freqs
    s = 1j * w

    H_analytical = np.ones_like(s, dtype=complex) * gain
    for z in zeros:
        H_analytical *= (s - z)
    for p in poles:
        H_analytical /= (s - p)

    ax_mag.semilogx(freqs, 20 * np.log10(np.abs(H_analytical)), label="Analytical Label", color="red", linestyle="--")
    ax_phase.semilogx(freqs, np.degrees(np.angle(H_analytical)), label="Analytical Label", color="red", linestyle="--")

    ax_mag.set_title(f"Bode Plot: {title}")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend()

    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both", alpha=0.3)

    mse = np.mean(np.abs(H_spice - H_analytical) ** 2)
    return mse


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Run the generator first or update DATASET_PATH.")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} circuits.")

    samples = random.sample(dataset, min(NUM_SAMPLES, len(dataset)))

    for i, data in enumerate(samples):
        poles = data["label"]["poles"]
        zeros = data["label"]["zeros"]
        gain = data["label"]["gain"]
        freqs = data["frequency_response"]["freqs"]
        H_spice = data["frequency_response"]["H_complex"]

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        ax_pz = fig.add_subplot(gs[:, 0])
        ax_mag = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1])

        plot_pole_zero_map(ax_pz, poles, zeros, f"PZ Map: {data['filter_type']}")

        mse = plot_bode(
            ax_mag,
            ax_phase,
            freqs,
            H_spice,
            poles,
            zeros,
            gain,
            data["filter_type"],
        )

        print(f"\nInspecting Sample {i + 1}: {data['filter_type']}")
        print(f"  ID: {data['id']}")
        print(f"  Poles: {poles}")
        print(f"  Zeros: {zeros}")
        print(f"  Gain: {gain:.2e}")
        print(f"  MSE vs SPICE: {mse:.2e}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
