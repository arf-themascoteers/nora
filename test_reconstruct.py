from reconstruct import Reconstructor


if __name__ == "__main__":
    ag = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    train = r"data/processed/47eb237b21511beb392f4845d460e399/train_random.csv"
    test = r"data/processed/47eb237b21511beb392f4845d460e399/test_random.csv"
    height, width = Reconstructor.recon(ag, save=False)
    Reconstructor.recon(train, height, width, save=False)
    Reconstructor.recon(test, height, width, save=False)

