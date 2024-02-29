import numpy as np
import tensorflow as tf

# 載入你的模型
model = tf.keras.models.load_model('your_model.h5')

# 指定圖片路徑
image_path = './1.jpg'

# 讀取圖片並進行前處理
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# 正規化圖片
img_array = img_array / 255.0

# 進行預測
predictions = model.predict(img_array)

# 取得預測結果
score = tf.nn.softmax(predictions[0])
class_names = ['rock', 'scissors', 'paper', 'other']
predicted_class = class_names[np.argmax(score)]

# 顯示結果
print("這張圖片被預測為：", predicted_class)
print("各類別的預測機率：", score.numpy())