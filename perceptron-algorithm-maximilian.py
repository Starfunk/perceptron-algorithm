import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv("perceptron_data.csv")

#The learning rate of the classifier
learning_rate = 0.01

#The number of iterations of gradient descent.
#TODO: Optimize number of rounds needed to train.
N = 10000

#The initial range of random numbers the coefficients, a,b,c can take on.
#The range is from 0 to the mean value of the x and y data points.
#TODO: Improve the range of the initial random numbers
a = random.uniform(0,df.mean()[0])

b = random.uniform(0, df.mean()[1])

c = random.uniform(0, df.mean()[1])

print("The initial random line is: "+str(round(a,3))+" * X1 + "+str(round(b,3))+" * X2 + "+str(round(c,3)))	

#x and y are the coordinates of the point. a,b, and c are the values of the line. 
def classify(x,y,label,a,b,c):
	value = x * a + y * b + c
	if value > 0:
		if label == 0:
			return True
		else:
			return False
			
	elif value < 0:
		if label == 1:
			return True
		else:
			return False
			
	elif value == 0:
		return False

def update_line(x,y,label,a,b,c):
	flag = classify(x,y,label,a,b,c)
	
	if flag == False and label == 1:
		a = a - learning_rate * x
		b = b - learning_rate * y
		c = c - learning_rate
		
	elif flag == False and label == 0:
		a = a + learning_rate * x
		b = b + learning_rate * y
		c = c + learning_rate
		
	return (a, b, c)

#The learning process:  
for i in range(N):
	rand = random.randint(0,df.shape[0]-1)
	points = df.iloc[rand,:]
	x = points[0]
	y = points[1]
	label = points[2]
	
	updated_coefficients = update_line(x,y,label,a,b,c)

	a = updated_coefficients[0]
	b = updated_coefficients[1]
	c = updated_coefficients[2]
	
print()	
print("After running " + str(N) + " iterations, the trained classifier is: "
	+str(round(a,3))+" * X1 + "+str(round(b,3))+" * X2 + "+str(round(c,3)))	
	

#Plotting the points!
x = df.iloc[:,0]
y = df.iloc[:,1]
label = df.iloc[:,2]
color=['red' if l == 1 else 'blue' for l in label]
ax = plt.axes()
plt.scatter(x,y,color=color)
ax.plot(x,-(a/b)*x - (c/b),color='black')

plt.show()




