import rasterio
import matplotlib.pyplot as plt

path = r"D:\src\nora\data\bands\TCI.tif"

with rasterio.open(path) as src:
    for i in range(1, src.count+1):
        data = src.read(i)
        print(data)
        print(data.max())
        print(data.min())
        plt.imshow(data, cmap="autumn")
        plt.colorbar()
        plt.title("2-D Heat Map")
        plt.show()
