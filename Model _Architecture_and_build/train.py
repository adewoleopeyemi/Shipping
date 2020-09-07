##Automatically assumes training is for deployment
from .build_model import build_nudity_detection_alogrithm
from preprocessing.preprocess import preprocess_multiple_images, train_test_split
import numpy as np

path_to_nonnudes = "Please Enter relative path to non nudes dataset contained in the dataset directory"
path_to_nudes = "Please Enter relativee path to nudes dataset contained in the dataset directory"

nonnudes = preprocess_multiple_images(path_to_directory_of_images=path_to_nonnudes)
nudes = preprocess_multiple_images(path_to_directory_of_images=path_to_nudes)

nonnudes_targets = np.zeros(len(nonnudes))
nudes_targets = np.ones(len(nudes))

X = np.concatenate((nonnudes, nudes))
y = np.concatenate((nonnudes_targets, nudes_targets))

((X_train, y_train), (X_test, y_train)) = train_test_split(X=X, y=y, percent_split=0.2)

model = build_nudity_detection_alogrithm(X=X_train, y=y_train)

if __name__ == "__main__":
    model.main()