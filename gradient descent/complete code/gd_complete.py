import pandas as pd
import matplotlib.pyplot as plt  

### Helper function use when needed
def plot_regression_line(X,m,b):
	regression_x = X.values
	regression_y = []
	for x in regression_x:
		y = m*x + b
		regression_y.append(y)

	plt.plot(regression_x,regression_y)
	plt.pause(1)


df = pd.read_csv("student_scores.csv")

X = df["Hours"]
Y = df["Scores"]

plt.plot(X,Y,'o')
plt.title("Implementing Gradient Descent")
plt.xlabel("Hours Studied")
plt.ylabel("Student Score")

#plt.show()

m = 0
b = 0


## takes in m,b and gives a better value of m,b 
## such that error reduces
def grad_desc(X,Y,m,b):

	for point in zip(X,Y):
		x = point[0]
		y_actual = point[1]

		y_prediction = m*x + b

		error = y_prediction - y_actual

		delta_m = -1 * (error*x) * 0.0005
		delta_b = -1 * (error) * 0.0005
		m = m + delta_m
		b = b + delta_b

	return m,b	 


for i in range(0,10):
	m,b = grad_desc(X,Y,m,b)
	plot_regression_line(X,m,b)

plt.show()


