from reconstruct import Reconstructor


if __name__ == "__main__":
    basedir = r"data/processed/47eb237b21511beb392f4845d460e399"
    f1 = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    f2 = r"data/processed/47eb237b21511beb392f4845d460e399/train_spatial.csv"
    f3 = r"data/processed/47eb237b21511beb392f4845d460e399/test_spatial.csv"
    height, width = Reconstructor.recon(f1)
    Reconstructor.recon(f2, height, width)
    Reconstructor.recon(f3, height, width)
