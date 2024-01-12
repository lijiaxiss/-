import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
Train_data = pd.read_csv(r'used_car_train_20200313.csv', sep=' ')
TestB_data = pd.read_csv(r'used_car_testB_20200421.csv', sep=' ')
#数据清洗
#将Train_data中的所有包含短横线（-）的元素替换为NaN（缺失值）
Train_data.replace(to_replace = '-', value = np.nan, inplace = True)
TestB_data.replace(to_replace = '-', value = np.nan, inplace = True)
#将Train_data中的所有NaN值用相应列的中位数进行填充
Train_data.fillna(Train_data.median(),inplace= True)
TestB_data.fillna(Train_data.median(),inplace= True)
#特征标签
tags = ['model','brand','bodyType','fuelType','regionCode','regionCode','regDate','creatDate','kilometer','notRepairedDamage','power','v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
       'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']

#修改异常数据
#将所有功率大于600的汽车的功率值都设置为600
Train_data['power'][Train_data['power']>600]=600
TestB_data['power'][TestB_data['power']>600]=600

#特征归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(Train_data[tags].values)
x = min_max_scaler.transform(Train_data[tags].values)
x_ = min_max_scaler.transform(TestB_data[tags].values)

#获得y值
y = Train_data['price'].values
#切分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
# #建模
# model = keras.Sequential([
#         keras.layers.Dense(250,activation='relu',input_shape=[26]),
#         keras.layers.Dense(250,activation='relu'),
#         keras.layers.Dense(250,activation='relu'),
#         keras.layers.Dense(1)])
model = keras.Sequential([
    # 输入层和第一个隐藏层。有256个神经元，使用ReLU（Rectified Linear Unit）作为激活函数。input_shape=[26]指定输入的形状是一个包含26个特征的一维数组
    keras.layers.Dense(256, activation='relu', input_shape=[26]),
    # 第二个隐藏层，有128个神经元，同样使用ReLU作为激活函数。
    keras.layers.Dense(128, activation='relu'),
    # 第三个隐藏层，有64个神经元，同样使用ReLU作为激活函数。
    keras.layers.Dense(64, activation='relu'),
    #输出层
    keras.layers.Dense(1)
    # keras.layers.BatchNormalization()

])
#在训练过程中调整模型权重以减小损失。指定学习率为0.1
optimizer = keras.optimizers.Adam(learning_rate=0.1)
#配置模型的损失函数和优化器。平均绝对误差（Mean Absolute Error）。
model.compile(loss='mean_absolute_error',
                optimizer='Adam')
#batch_size=256表示每次更新权重时使用的样本数为256，epochs=30表示整个训练数据集将被遍历30次。在每个训练周期结束时使用验证数据来评估模型性能
history =model.fit(x_train,y_train,batch_size =256,epochs=30,validation_data=(x_test, y_test))

#输出结果预测
y_=model.predict(x_)
data_test_price = pd.DataFrame(y_,columns = ['price'])
results = pd.concat([TestB_data['SaleID'],data_test_price],axis = 1)
results.to_csv('results.csv',sep = ',',index = None)

# 可视化训练过程中的损失值
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
plot_loss(history)

