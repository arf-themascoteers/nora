import torch
from ann import ANN
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_ds, test_ds, algorithm):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if algorithm == "ann":
            model = ANN(device, train_ds, test_ds)
            model.train_model()
            return model.test()
        else:
            train_x = train_ds.x
            train_y = train_ds.y
            test_x = test_ds.x
            test_y = test_ds.y

            model_instance = None
            if algorithm == "mlr":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()

            model_instance = model_instance.fit(train_x, train_y)
            return model_instance.score(test_x, test_y)