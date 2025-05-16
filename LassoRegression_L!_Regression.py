import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from pathlib import Path
import urllib.request as request

from ConstantsModule import *  # Make sure this defines Dataset_Link

# Step 1: Configuration class for downloading data
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


# Step 2: Config object
config = DataIngestionConfig(
    root_dir=Path("Dataset"),
    source_URL=Dataset_Link,
    local_data_file=Path("Dataset/Experience-Salary.csv"),
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]
)


def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")


# Step 3: Lasso Regression Implementation
class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penalty, verbose=False):
        download_project_file(config.source_URL, config.local_data_file)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.verbose = verbose

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
            if self.verbose and i % 100 == 0:
                loss = np.mean((self.Y - self.predict(self.X))**2)
                print(f"Iteration {i}: Loss = {loss:.4f}")
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        error = self.Y - Y_pred

        # Vectorized gradient calculation
        dW = (-2 * self.X.T.dot(error) + self.l1_penalty * np.sign(self.W)) / self.m
        db = -2 * np.sum(error) / self.m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        return self

    def predict(self, X):
        return X.dot(self.W) + self.b


# Step 4: Main function
def main():
    df = pd.read_csv("Dataset/Experience-Salary.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

    model = LassoRegression(learning_rate=0.01, iterations=1000, l1_penalty=10, verbose=True)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # Evaluation metrics
    print("\n📊 Evaluation Metrics:")
    print("Mean Squared Error (MSE):", round(mean_squared_error(Y_test, Y_pred), 2))
    print("R² Score:", round(r2_score(Y_test, Y_pred), 2))

    print("\n🧠 Model Parameters:")
    print("Predicted values:", np.round(Y_pred[:3], 2))
    print("Real values:     ", Y_test[:3])
    print("Trained W:       ", round(model.W[0], 2))
    print("Trained b:       ", round(model.b, 2))

    # Only visualize for 1D features
    if X.shape[1] == 1:
        plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
        plt.plot(X_test, Y_pred, color='orange', label='Lasso Regression Line')
        plt.title('Salary vs Experience (Lasso Regression)')
        plt.xlabel('Years of Experience (Standardized)')
        plt.ylabel('Salary')
        plt.legend()
        plt.show()
    else:
        print("⚠️ Skipping plot: Visualization is only implemented for 1D feature data.")


if __name__ == "__main__":
    main()



