from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import numpy as np

(_, _), (x_test, y_test) = cifar10.load_data()


model = load_model('cifar10_ipc50_result.h5')

predictions = model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test.flatten()
accuracy = np.mean(predicted_labels == true_labels)
print(f"CTA: {accuracy}")
