import rasterio
import matplotlib.pyplot as plt

path = r"D:\Data\Sentinel-2\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA\R60m\T54HXE_20220423T002659_AOT_60m.jp2"

with rasterio.open(path) as src:
    data = src.read(1)
    print(data)
    print(data.max())
    print(data.min())
    plt.imshow(data, cmap="autumn")
    plt.colorbar()
    plt.title("2-D Heat Map")
    plt.show()
