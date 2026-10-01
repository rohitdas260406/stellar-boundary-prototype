import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
X_features, y_labels = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
model = LogisticRegression()
model.fit(X_features, y_labels)
print("Star Center:", X_features[y_labels == 0].mean(axis=0))
print("Galaxy Center:", X_features[y_labels == 1].mean(axis=0))
unseen_object = np.array([[-2.5, 9.0]])
prediction = model.predict(unseen_object)
if prediction[0]==1:
    print("The Machine says: Galaxy")
else:
    print("The Machine says: Star")
print(f"Mathematical Boundary Weights: {model.coef_}")
np.savetxt("stars.dat", X_features[y_labels == 0], fmt="%.4f")
np.savetxt("galaxies.dat", X_features[y_labels == 1], fmt="%.4f")
np.savetxt("test_point.dat", unseen_object, fmt="%.4f")
slope = -model.coef_[0][0] / model.coef_[0][1]
intercept = -model.intercept_[0] / model.coef_[0][1]
print(f"GNUPLOT Equation: f(x) = {slope:.4f} * x + ({intercept:.4f})")
