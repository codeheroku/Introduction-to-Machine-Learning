import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


faces_image = np.load('olivetti_faces.npy')
faces_target = np.load('olivetti_faces_target.npy')

n_row = 64
n_col = 64

faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1] * faces_image.shape[2])

plt.imshow(faces_image[22],cmap='gray')

plt.show()

n_samples = faces_image.shape[0]
X = faces_data
n_features = faces_data.shape[1]
# the label to predict is the id of the person
y = faces_target
n_classes = faces_target.shape[0]
#merged.to_csv("output.csv", index=False)

df= pd.DataFrame(X)
df['target'] = pd.Series(y, index=df.index)

df.to_csv("face_data.csv", index=False)
