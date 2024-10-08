import pickle
import numpy as np  

pipe = pickle.load(open(r'M:\Machine Learning\Notes ML\EDA\Feature Engineering\Pipelines\models\titanic_pipe.pkl', 'rb'))
# Define test input: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
test_input = np.array([1, 0, 21, 0, 0, 20, 1.0], dtype=float).reshape(1, 7)

predicted_output = pipe.predict(test_input)

if 0 in predicted_output:
    print('Not Rescued')
else:
    print('Rescued')
