import torch
from ann import ANN
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_ds, test_ds, algorithm):
        train_x = train_ds.x
        train_y = train_ds.y
        test_x = test_ds.x
        test_y = test_ds.y
        y_hats = None
        if algorithm == "ann":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANN(device, train_ds, test_ds)
            model_instance.train_model()
            y_hats = model_instance.test()
        else:
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
            y_hats = model_instance.predict(test_x)

        rmse = mean_squared_error(y_hats, test_y, squared=False)
        r2 = r2_score(y_hats, test_y)
        return r2, rmse