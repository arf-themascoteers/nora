import rasterio
import matplotlib.pyplot as plt

b02 = r"D:\Data\Sentinel-2\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA\R60m\T54HXE_20220423T002659_B04_60m.jp2"
tci = r"D:\Data\Sentinel-2\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA\R60m\T54HXE_20220423T002659_TCI_60m.jp2"


def read(src):
    with rasterio.open(src) as src:
        print(src.count)
        for i in range(src.count):
            data = src.read(i+1)
            print(data)
            print(data.max())
            print(data.min())
            plt.imshow(data, cmap="autumn")
            plt.colorbar()
            plt.title("2-D Heat Map")
            plt.show()


if __name__ == "__main__":
    read(tci)
    read(b02)