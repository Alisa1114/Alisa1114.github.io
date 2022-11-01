---
layout: post  
title:  "用Numpy建構深度神經網路"  
date:   2022-11-01 11:12:53  
math: true
---  
進入研究所後，我上了一堂叫深度學習與人工神經網路的課程，原本我明明是抱著拿營養學分的態度來上課的，結果老師第一個作業就是要我們手刻一個神經網路...

但是沒有關係，剛好我幾年前就在Coursera學~~廢~~了手刻模型，但為了符合老師教材，下面方法我還是照著老師的方法做出來。


---

#### Import需要的library


```python
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
import cv2
import os
from os import listdir
from os.path import isfile, join
```

---

#### 讀取dataset並回傳圖片跟label的函式
* label_folders: 要讀取的labels
* dataset_path: 要讀取的資料集路徑
* preprocessing: 是否要做一些圖片的前處理，主要是把圖片轉成1D vector並normalize到0~1之間
* encoding: 是否要對labels做one hot encoding


```python
def dataset(label_folders, dataset_path='MNIST/train', preprocessing=True, encoding=True):
  img_list = []
  label_list = []
  enc = OneHotEncoder(handle_unknown='ignore')
  label_folders = np.array(label_folders).reshape(-1, 1)
  enc.fit(label_folders)
  
  # print(enc.categories_)

  for l in label_folders:
    labelPath = join(dataset_path, str(l.item()))
    file_names = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

    for file_name in file_names:
      img_path = join(labelPath, file_name)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

      if preprocessing:
        img = cv2.rotate(img, cv2.ROTATE_180) # 不知為何助教給的圖片轉了180度，所以在這裡轉回來
        img = img.reshape(-1).astype('float32')
        img /= 255.0
        
      l_enc = l  
      if encoding:
        # print(l)
        l_enc = enc.transform([l]).toarray()[0]
        # print(l_enc)
      
      img_list.append(img)
      label_list.append(l_enc)

  return np.array(img_list), np.array(label_list)

def test_dataset(dataset_path, preprocessing=True):
  def takeSecond(elem):
    return elem[1]
  
  data_list = []
  
  file_names = [f for f in listdir(dataset_path)]
  file_names.sort()
  
  for file_name in file_names:
    img_path = join(dataset_path, file_name) # 'mnist_testData/Tesing_data/0000.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if preprocessing:
      img = cv2.rotate(img, cv2.ROTATE_180)
      img = img.reshape(-1).astype('float32')
      img /= 255.0
    
    data_list.append((img, file_name))
    # data_list.sort(key=takeSecond)
    
  return data_list
```

---

#### 讀取dataset，讓它回傳數字2, 4, 5, 6, 7的圖片和labels
* images shape: [number of images, height * width]
* labels shape: [number of labels, number of classes]


```python
classes = [2, 4, 5, 6, 7]
training_images, training_labels = dataset(classes, dataset_path='Training_data')
# valid_images, valid_labels = dataset(classes, dataset_path='MNIST/valid', encoding=False)
```

---

#### $$Sigmoid(x) = \frac{1}{1 + \exp^{-x}}$$


```python
def sigmoid(x):
  # return 1.0 / (1.0 + np.exp(-x))
  return expit(x) # avoid overflow
```

---

#### 搭建model為一個class
* feed_forward: 做feed forward
$$a_j = f(\sum_{i} w_{ji}a_{i})$$
* train_model: 丟入dataset和超參數來訓練模型
* update: 利用back propagation更新模型
$$\Delta w_{ji}(n) = \alpha\Delta w_{ji}(n-1) + \eta\delta_ja_i$$
* back_propagation: 計算\\(\Delta w\\)和\\(\Delta b\\)
$$\Delta w_{ji}(n) = \alpha\Delta w_{ji}(n-1) + \eta\delta_ja_i$$
* predict: 預測圖片的label值
* compute_cost: 計算error
$$\delta_j = 
\begin{cases}
(d_j - a_j)a_j(1 - a_j) & \text{, if } j \text{ is an output unit} \\
a_j(1 - a_j)\sum_{k=1}^{n(n\in h+1)}\delta_{k,h+1}w_{kj,h+1}, & \text{, if } j \text{ is an hidden unit}
\end{cases}
$$


```python
from tqdm.auto import tqdm

class neural_network():
  def __init__(self, sizes):
    """suppose sizes=[2, 3, 4], weights would be [3, 2], [4, 3], biases would be [3, 1], [4, 1]"""
    self.sizes = sizes
    self.num_layer = len(self.sizes)
    self.weights = [np.random.randn(self.sizes[i], self.sizes[i-1]) for i in range(1, self.num_layer)]
    self.biases = [np.random.randn(self.sizes[i], 1) for i in range(1, self.num_layer)]
    self.last_delta_w = [np.zeros((self.sizes[i], self.sizes[i-1])) for i in range(1, self.num_layer)]
    self.last_delta_b = [np.zeros((self.sizes[i], 1)) for i in range(1, self.num_layer)]
    self.momentum = 0.0
    
  def feed_forward(self, a):
    for w, b in zip(self.weights, self.biases):
      a = sigmoid(np.dot(w, a) + b)
      
    return a
  
  def train_model(self, images, labels, num_epoch, learning_rate, batch_size, valid_images=None, valid_labels=None, momentum=0.0):
    n = len(images)
    self.momentum = momentum
    
    for epoch in tqdm(range(num_epoch)):
      
      images, labels = shuffle(images, labels)
     
      mini_batches_images = [images[i:i+batch_size] for i in range(0, n, batch_size)]
      mini_batches_labels = [labels[i:i+batch_size] for i in range(0, n, batch_size)]
      
      for image, label in zip(mini_batches_images, mini_batches_labels):
        self.update(learning_rate, image,label)
      
      # Calculate test data accuracy
      if valid_images is not None:
        acc = 0.0
        for image, label in zip(valid_images, valid_labels):
          result = self.predict(image)
          if classes[result] == label.item():
            acc += 1
        print('Epoch {} / {}     Accuracy : {}'.format(epoch+1, num_epoch, acc/len(valid_images)))
  
  def update(self, learning_rate, image, label):
    
    delta_weight, delta_bias = self.back_propagation(image, label)
    
    self.weights = [w + learning_rate * delta_w/len(image) for w, delta_w in zip(self.weights, delta_weight)]
    self.biases = [b + learning_rate * delta_b/len(image) for b, delta_b in zip(self.biases, delta_bias)] 
    
    self.last_delta_w = [w for w in delta_weight]
    self.last_delta_b = [b for b in delta_bias]       
      
  
  def back_propagation(self, image, label):
    delta_weights = [np.zeros(w.shape) for w in self.weights]
    delta_biases = [np.zeros(b.shape) for b in self.biases]
    
    activations = [image.T]
    activation = image.T
    
    for w, b in zip(self.weights, self.biases):
      activation = sigmoid(np.dot(w, activation) + b)
      activations.append(activation)
      
    delta = self.compute_cost(label.T, activations)
    
    for i, (a_i, delta_j) in enumerate(zip(activations, delta)):
      delta_weights[i] = self.momentum * self.last_delta_w[i] + delta_weights[i] + np.dot(delta_j, a_i.T)
      delta_biases[i] = self.momentum * self.last_delta_b[i] + delta_biases[i] + delta_j.sum(axis=-1, keepdims=True)
        
    return delta_weights, delta_biases
  
  def predict(self, test_data):
    # test_data: [784]
    y = self.feed_forward(test_data[...,None]) # [784, 1] -> [10, 1]
    results = np.argmax(y, axis=0) # [0.1, 0.3, 0.2, 0.4, 0.9, 0.4, 0.3, 0.4, 0.1, 0.5] -> [4]
  
    return results.item() # 4
    
  def compute_cost(self, label, activation):
    delta = []
    w = self.weights[::-1]
    for i, a in enumerate(activation[-1:0:-1]): #從最後一個往回取到index=1
      
      if i == 0:
        delta.append((label - a) * a * (1.0 - a))   
      else:
        delta.append(a * (1.0 - a) * (np.dot(w[i-1].T, delta[i-1])))
    
    return delta[::-1]    
```

---

#### 初始化並訓練模型


```python
model = neural_network([training_images.shape[-1], 1024, 256, 64, len(classes)])
model.train_model(training_images, 
                  training_labels, 
                  num_epoch=50, 
                  learning_rate=0.1, 
                  batch_size=8,
                  momentum=0.05)
```

#### 儲存模型權重
~~但我自己沒測試過能不能用就是了~~


```python
import pickle

checkpoint = {}
checkpoint['weights'] = model.weights
checkpoint['biases'] = model.biases

with open('checkpoint.pickle', 'wb') as f:
  pickle.dump(checkpoint, f)
```

---

#### 預測助教給的test set，並將結果寫進txt檔


```python
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
  
data_list = test_dataset('Testing_data1') # (image, file_name) * length of list

with open('student_ID.txt', 'w') as f:
  for img, file_name in data_list:
    result = model.predict(img)
    print('{} {}\n'.format(file_name.split('.')[0], classes[result]))
    f.write('{} {}\n'.format(file_name.split('.')[0], classes[result]))
```
