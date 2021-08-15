import sklearn.datasets as datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.tree.export import export_text

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image  
import pydotplus

# Loading dataset and formulating the dataframe.

# Loading dataset
iris_data = datasets.load_iris()

# Creating dataframe
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Adding target values as a column in dataframe
df['target'] = iris_data.target

# Printing dataframe head
print(df.head(15))

#Printing dataframe information
print("Dataset Information -->\n")
print(df.info())
print("\n"*2)

#Printing target values of dataset
target = iris_data.target
print("All target values -->\n")
print(target)
print("\n"*2)

#Printing unique target values of dataset
unique = df['target'].unique()
print("Unique target values -->\n")
print(unique)
print("\n"*2)

#Checking the presence of NULL values
null_check = df.isnull().sum()
print("Null values -->\n")
print(null_check)
print("\nAs clearly seen above, there are no null values and thus, there's no need to process it.")
print("\n"*2)

#Printing mathematical dataset analysis
print("Rigorous mathematical analysis of dataset -->\n")
print(df.describe())

#Visulaizing the dataframe.

#Printing visual plots for dataframe
print("Visualizing dataframe -->\n")
plot_var_1 = sns.pairplot(df, kind = 'scatter', diag_kind='kde', hue = 'target', palette=["C0", "C1", "C2"])
plot_var_2 = plot_var_1.map_lower(sns.kdeplot, levels=3)
plot_1 = plot_var_2.map_upper(sns.kdeplot, levels=3)
print(plot_1)
print("\n"*2)

#Visualizing the correlation among features.

#Checking correlation among features
print("Visualizing correlation between features using heatmap -->\n")
plot_2 = sns.heatmap(df.corr(method='pearson'), cmap='Reds')
print(plot_2)

#Visualizing the distribution of target values via Pie Chart.

#Printing pie chart for target value distribution
print("Pie Chart for all target values -->\n")
temp_var_1 = df['target'].value_counts()
plt.pie(temp_var_1.values, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-verginica'], explode = (0.05,0.05,0.05), autopct='%1.2f%%', shadow=True)
plt.xlabel('Target values')
plt.show()

# Getting the X and Y dataset ready.

#Segregating dataset into input and output (X and Y)
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
Y = df[['target']]
print("The X dataset -->\n")
print(X)
print("\n"*2)
print("The Y dataset -->\n")
print(Y)

#Splitting the datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.20)
print("The X training dataset -->\n")
print(X_train)
print("\n"*2)
print("The X test dataset -->\n")
print(X_test)
print("\n"*2)
print("The Y train dataset -->\n")
print(Y_train)
print("\n"*2)
print("The Y test dataset -->\n")
print(Y_test)
print("\n"*2)

#Printing dataset sizes
print("Dataset sizes -->\n")
print('X Training data size  : ',X_train.size, "values")
print('X Test data size      : ',X_test.size, "values")
print('Y Training data size  : ',Y_train.size, "values")
print('Y Test data size      : ',Y_test.size, "values")

#Creating model
dt_model = DecisionTreeClassifier(max_depth = 10, random_state = 1)
#Fitting the model
dt_model.fit(X_train,Y_train)
print("Decision Tree created !")

#Making presictions
Y_pred = dt_model.predict(X_test)

#Using various evaluation parameters
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test,Y_pred)
accuracy = dt_model.score(X_test,Y_test)
print("Confusion Matrix -->\n")
print(conf_matrix)
print("\n"*2)
print("Classification Report -->\n")
print(class_report)
print("\n"*2)
print("Model Score-->\n")
print(accuracy)
print()
print("This signifies that the model's accuracy is :", (str(format(accuracy*100, '.2f')) + "%"), "!")

#Making prediction on custom values.

#Defining a function to return message reagarding predicted output speicies 
def species(num):
  if (num == 0):
    return "The predicted species for custom input is Iris-setosa !"
  elif (num == 1):
    return "The predicted species for custom input is Iris-versicolor !"
  elif (num == 2):
    return "The predicted species for custom input is Iris-virginica !"
  else:
    print("Prediction Error !")

#Making custom prediction
custom_pred = species(dt_model.predict([[5.9,3.0,5.1,1.8]]))
print(custom_pred)

#Printing graphical visual of decision tree
print("Visual Graphic of Decision Tree model -->\n")
dot_data = StringIO()
export_graphviz(dt_model, out_file=dot_data, filled = True, rotate = False, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#Printing text visual of decision tree
text_visual = export_text(dt_model,feature_names=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'))
print("Text Visual of decision tree -->\n")
print(text_visual)

#Thank you!
