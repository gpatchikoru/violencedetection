import argparse
import h5py
import matplotlib.pyplot as plt

def plot_counts(h5_path: str, out_png: str):
   
    with h5py.File(h5_path, 'r') as f:
        labels = [grp.attrs['label'] for grp in f.values()]
    non_violence = labels.count(0)
    violence     = labels.count(1)

    # Plot
    plt.figure(figsize=(6,4))
    plt.bar(['NonViolence','Violence'], [non_violence, violence])
    plt.title('Video Clip Counts by Label')
    plt.ylabel('Number of Clips')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved bar chart to {out_png}: NonViolence={non_violence}, Violence={violence}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot clip‐count bar chart from HDF5")
    parser.add_argument('--h5',     required=True, help="Path to violence.h5")
    parser.add_argument('--out',    required=True, help="Output PNG filename")
    args = parser.parse_args()
    plot_counts(args.h5, args.out)
