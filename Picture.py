# person_recognition.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class PersonRecognitionModel:
    def __init__(self, target_size=(128, 128), batch_size=32):
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_indices = None
        self.model = self.build_model()
        
    def build_model(self):
        """构建卷积神经网络模型"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.target_size, 3)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.get_example_classes()), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def get_example_classes(self):
        """为模型构建提供示例类别（实际训练时会替换）"""
        return ['class1', 'class2']
    
    def prepare_data(self, data_dir):
        """准备训练和验证数据集"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20%数据用作验证集
        )
        
        # 训练集
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # 验证集
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_indices = train_generator.class_indices
        print(f"Detected classes: {list(self.class_indices.keys())}")
        
        # 更新模型最后一层以适应实际类别数
        num_classes = len(self.class_indices)
        self.model.pop()  # 移除最后一层
        self.model.add(layers.Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        return train_generator, validation_generator
    
    def train(self, data_dir, epochs=20, save_path='person_recognition_model.h5'):
        """训练模型"""
        train_gen, val_gen = self.prepare_data(data_dir)
        
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // self.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // self.batch_size
        )
        
        self.model.save(save_path)
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """绘制训练历史图表"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.savefig('training_history.png')
        plt.show()
    
    def predict(self, image_path, model_path='person_recognition_model.h5'):
        """使用训练好的模型进行预测"""
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            model = self.model
        
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        
        if self.class_indices:
            class_labels = {v: k for k, v in self.class_indices.items()}
            return class_labels[predicted_class]
        return f"Class_{predicted_class}"

if __name__ == "__main__":
    # 创建模型实例
    recognizer = PersonRecognitionModel()
    
    # 训练配置
    DATA_DIR = "dataset"  # 包含人物子文件夹的目录
    MODEL_PATH = "person_model.h5"
    
    # 检查数据目录是否存在
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        print(f"Training model with data from {DATA_DIR}...")
        recognizer.train(DATA_DIR, epochs=20, save_path=MODEL_PATH)
        print("Training complete!")
        
        # 示例预测
        TEST_IMAGE = os.path.join(DATA_DIR, "sample_image.jpg")
        if os.path.exists(TEST_IMAGE):
            result = recognizer.predict(TEST_IMAGE, MODEL_PATH)
            print(f"Prediction for {TEST_IMAGE}: {result}")
    else:
        print("Dataset directory not found. Creating sample directory structure...")
        os.makedirs(os.path.join(DATA_DIR, "person1"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "person2"), exist_ok=True)
        print(f"Created sample directory structure at {DATA_DIR}.")
        print("Please add your training images in subfolders like:")
        print(f"- {DATA_DIR}/person1/")
        print(f"- {DATA_DIR}/person2/")
        print("Then run the script again to train the model.")