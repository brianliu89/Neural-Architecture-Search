import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, InputLayer, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

data = np.load('cifar10_ipc50_simple_result_seed0.npy', allow_pickle=True).item()

images = data['condensed_dataset']['x']  # Extract graph data
labels = data['condensed_dataset']['y']  # Extract graph labels

images = np.asarray(images)
labels = np.asarray(labels)

# class CNN(nn.model)
#     def __init__(self):
#         super(CNN,self).__init__()

#         self.layers = nn.Sequential(
#             nn.Conv2d(3,32,3,padding=1),
#             nn.batchNorm2d(32),
#             nn.relu(),
#             nn.MaxPooling2D(2),
            
#             nn.Conv2D(32,64,3,padding=1),
#             nn.batchNorm2d(64),
#             nn.relu(),
#             nn.MaxPooling2D(2),

#             nn.Conv2D(64,128,3,padding=1),
#             nn.batchNorm2d(128),
#             nn.relu(),
#         )

#         self.fc = nn.Linear(8192,10)

train_batch_size = 32
n_epochs = 50
lr = 0.01

search_space = {
    'num_conv_layers': [3],
    'num_dense_layers': [2],
    'conv_layer_size': [128],
    'batch_norm_size': [32],
    'dense_layer_size': [128],
    'kernel_size': [(3, 3)],
    'activation': ['relu'],
    'dropout_rate': [0.0],
}

def random_search(search_space, num_samples):
    models = []
    for _ in range(num_samples):
        model = {k: random.choice(v) for k, v in search_space.items()}
        models.append(model)
    return models


sampled_models = random_search(search_space, 10)


def build_cnn_model(model_architecture, input_shape, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    for _ in range(model_architecture['num_conv_layers']):
        model.add(Conv2D(model_architecture['conv_layer_size'], kernel_size=model_architecture['kernel_size'], activation=None, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(model_architecture['activation']))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for _ in range(model_architecture['num_dense_layers']):
        model.add(Dense(model_architecture['dense_layer_size'], activation=model_architecture['activation']))
        if model_architecture['dropout_rate'] > 0:
            model.add(Dropout(model_architecture['dropout_rate']))

    model.add(Dense(num_classes, activation='softmax'))
    return model


input_shape = images.shape[1:]
num_classes = labels.shape[1]

best_accuracy = 0.0
best_model = None
best_model_arch = None

for model_arch in sampled_models:
    model = build_cnn_model(model_arch, input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=n_epochs, batch_size=train_batch_size, verbose=1)

    test_loss, test_accuracy = model.evaluate(images, labels, batch_size=train_batch_size)
    print(f"Model: {model_arch}, Test Accuracy: {test_accuracy}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = model
        best_model_arch = model_arch
        best_model.save('cifar10_ipc50_simple.h5')

print(f"Best Model Architecture: {best_model_arch}")
print(f"Best Test Accuracy: {best_accuracy}")