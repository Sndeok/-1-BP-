import numpy
import scipy.special
import os
from PIL import Image
#创建神经网络类，以便于实例化成不同的实例
class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
    	#初始化输入层、隐藏层、输出层的节点个数、学习率
        self.inodes=input_nodes
        self.hnodes=hidden_nodes
        self.onodes=output_nodes
        #定义输入层与隐藏层之间的初始权重参数 随机复制权重
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #定义隐藏层与输出层之间的初始权重参数  随机复制权重
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learning_rate
        #定义激活函数sigmoid
        self.activation_function=lambda x: scipy.special.expit(x)
        pass

    def train(self,input_list,target_list):
    	# 输入数据矩阵、目标结果矩阵
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T
        # 隐藏层输入
        hidden_inputs=numpy.dot(self.wih,inputs)
        # 隐藏层激活后输出
        hidden_outputs=self.activation_function(hidden_inputs)
        # 最终输入
        final_inputs=numpy.dot(self.who,hidden_outputs)
        # 最终激活后输出
        final_outputs=self.activation_function(final_inputs)
        #计算误差
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #更新迭代初始权重，公式为权重更新公式，原理为导数、梯度下降。
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),(numpy.transpose(inputs)))
        pass
    #查询函数，相当于sklearn中的predict功能，预测新样本的种类
    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs



input_nodes =784 #输入层的数量，取决于输入图片(训练和识别)的像素，像素点的数量等于输入层的数量
hidden_nodes =200 #隐藏层的数量，一般比输入层少，但具体不确定，可根据准确率进行调整
output_nodes =10 #输出层的数量，等于需要分类的数量。

learning_rate =0.1 #定义学习率

#======================训练=======================

#用我们的类创建一个神经网络实例
n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#读取数据
training_data_file=open('mnist_train.csv','r',encoding='UTF-8')
training_data = training_data_file.readline()
training_data_list = [training_data]
while training_data_list is not None and training_data_list != '':
    if training_data != '':
    	training_data = training_data_file.readline()
    	training_data_list.append(training_data)
    else:
        break
training_data_file.close()
training_data_list.pop(-1)


accuracy_list = []

#训练数据集用于训练的次数
epochs=2
for e in range(epochs):
    for record in training_data_list:
    	#根据逗号，将文本数据进行拆分
        all_values=record.split(',')
        #将文本字符串转化为实数，并创建这些数字的数组。
        inputs=(numpy.asfarray(all_values[1:])/255*0.99+0.01)
        #创建用零填充的数组，数组的长度为output_nodes,加0.01解决了0输入造成的问题
        targets=numpy.zeros(output_nodes)+0.01
        #使用目标标签，将正确元素设置为0.99
        targets[int(all_values[0])]=0.99

        #导入训练网络更新权重值
        n.train(inputs,targets)
        pass
    pass

#=============================测试=======================
test_data_file=open('mnist_test.csv','r',encoding='UTF-8')
test_data = test_data_file.readline()
test_data_list = [test_data]
while test_data_list is not None and test_data_list != '':
    if test_data != '':
    	test_data = test_data_file.readline()
    	test_data_list.append(test_data)
    else:
        break
test_data_file.close()
test_data_list.pop(-1)
#通过类方法query输出test数据集中的每一个样本的训练标签和实例标签进行对比。
scorecord=[]
for record in test_data_list:
	#根据逗号，将文本数据进行拆分，csv格式是纯文本
    all_values=record.split(',')
    #获得标签
    correct_label=int(all_values[0])
    #将文本字符串转化为实数，并创建这些数字的数组。
    inputs=(numpy.asfarray(all_values[1:])/255*0.99)+0.01
    #导入查询函数获得最终输出
    outputs=n.query(inputs)
    #转换成标签值
    label=numpy.argmax(outputs)

    if  (label==correct_label):
        scorecord.append(1)
    else:
        scorecord.append(0)
        pass
    pass
#计算准确率
scorecord_array=numpy.asarray(scorecord)
print("使用mnist_test.csv测试神经网络 准确度=",scorecord_array.sum()/scorecord_array.size)

#######################图片数据测试#############################
# 获取文件夹中的所有PNG图像文件
image_folder = "E:\Code\Python\智能信息处理实验\测试图片"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.PNG')]

# 遍历测试图像
for image_file in image_files:
    # 打开图像文件
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    # 将图像转换为灰度
    gray_image = image.convert('L')
    # 将灰度图像数据转化为一维列表
    pixel_data = list(gray_image.getdata())
    # 使用神经网络进行识别
    output = n.query(pixel_data)
    # 找到神经网络输出中最高值的索引，即识别的数字
    recognized_digit = numpy.argmax(output)
    # 输出识别结果
    print(f"{image_file} 的识别结果是：{recognized_digit}")
