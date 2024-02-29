import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import pathlib

Keras = tf.keras
layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

# 檔案路徑
data_dir = pathlib.Path('./photos')

# 輸出圖片檔案總數量
img_count = len(list(data_dir.glob('*/*.jpg')))
print(img_count)

# 資料
rock = list(data_dir.glob('rock/*'))
PIL.Image.open(str(rock[0]))
PIL.Image.open(str(rock[1]))

scissors = list(data_dir.glob('scissors/*'))
PIL.Image.open(str(scissors[0]))
PIL.Image.open(str(scissors[1]))

paper = list(data_dir.glob('paper/*'))
PIL.Image.open(str(paper[0]))
PIL.Image.open(str(paper[1]))

other = list(data_dir.glob('other/*'))
PIL.Image.open(str(other[0]))
PIL.Image.open(str(other[1]))

# 參數設定
batch_size = 32
img_height = 224
img_width = 224

# 驗證拆分
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# 數據集類別名稱
class_names = train_ds.class_names
print(class_names)

# 训练数据集中的前 9 个图像
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
  
# 顯示訓練的圖片
plt.show()

# 数据集传递
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# 缓冲预提取
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 數據標準化控制在 0-1
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

# 創建模型
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# 編譯模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 查看網路層
model.summary()

# 訓練模型
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 呈現結果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 輸出模型
# epochs = 10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

model.save("your_model.h5")