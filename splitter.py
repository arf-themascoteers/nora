class Splitter:
    def __init__(self, csv, mode="random"):
        self.csv = csv
        #self.config_list = config_list

    def split(self):
        return None, None


if __name__ == "__main__":
    import reconstruct
    f = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    s = Splitter(f, mode="spatial")
    train, test = s.split()
    reconstruct.recon(train)
    reconstruct.recon(test)
