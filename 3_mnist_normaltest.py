import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import imageio
import glob

class neuralNetwork:
	def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
		# 设置每层的节点
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		# 链接权重矩阵
		self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
		self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
		# 学习率
		self.lr = learningrate
		
		# 激活函数为匿名函数
		self.activation_function = lambda x:scipy.special.expit(x)
		pass

	#训练构建的神经网络
	def train(self,inputs_list,targets_list):
		# 将输入数组转为二维矩阵
		inputs = np.array(inputs_list,ndmin=2).T
		targets = np.array(targets_list,ndmin=2).T

		# 计算隐藏层内的信息
		hidden_inputs = np.dot(self.wih, inputs)
		# 计算隐藏层的输出信息
		hidden_outputs = self.activation_function(hidden_inputs)
		# 计算最后输出层的信息
		final_inputs = np.dot(self.who,hidden_outputs)
		# 计算最后输出层产生的信息
		final_outputs = self.activation_function(final_inputs)
		# 输出层的误差为目标值与实际值之差
		output_errors = targets - final_outputs
		# 隐藏层误差即为输出误差
		hidden_errors = np.dot(self.who.T,output_errors)

		#更新隐藏层和输出层之间的权重
		self.who += self.lr * np.dot((output_errors * final_outputs * 
						(1.0 - final_outputs)),np.transpose(hidden_outputs))

		#更新输入层和隐藏层之间的权重
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * 
						(1.0 - hidden_outputs)),np.transpose(inputs))
		pass

	def query(self,inputs_list):
		# 将输入列表转换为二维矩阵
		inputs = np.array(inputs_list,ndmin=2).T
		# 计算隐藏层内的信息
		hidden_inputs = np.dot(self.wih, inputs)
		# 计算隐藏层的输出信息
		hidden_outputs = self.activation_function(hidden_inputs)
		# 计算最后输出层的信息
		final_inputs = np.dot(self.who,hidden_outputs)
		# 计算最后输出层产生的信息
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
		
input_nodes = 784 # 28 * 28 = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("my_mnist_data/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1
for e in range(epochs):
	for record in training_data_list:
		# 将数据按','分开
		all_values = record.split(',')
		# 缩放并移动输入数据
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		# 创建target输出值
		targets = np.zeros(output_nodes) + 0.01
		# all_values[0]是数据的目标下标
		targets[int(all_values[0])] = 0.99
		n.train(inputs,targets)

our_own_dataset = []

for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
	print ("loading ... ", image_file_name)
	label = int(image_file_name[-5:-4])
	img_array = imageio.imread(image_file_name, as_gray=True)
	img_data  = 255.0 - img_array.reshape(784)
	img_data = (img_data / 255.0 * 0.99) + 0.01
	print(np.min(img_data))
	print(np.max(img_data))
	plt.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')
	#plt.show()
	outputs = n.query(img_data)
	print (outputs)
	label = np.argmax(outputs)
	print("network says",label)
	pass

