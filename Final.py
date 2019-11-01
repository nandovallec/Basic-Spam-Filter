import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

data = pd.read_csv("new_sms_train.csv", error_bad_lines=False, delimiter='\t')

f = feature_extraction.text.CountVectorizer(stop_words='english')
X = f.fit_transform(data["v2"])


data["v1"] = data["v1"].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.32, random_state=42)


list_alpha = np.arange(1 / 100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test = np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count] = bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data=matrix,columns=['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
(models.head(n=5))

best_index = models['Test Accuracy'].idxmax()
models.iloc[best_index, :]
(models[models['Test Accuracy']==models['Test Accuracy'].max()])

models[models['Test Accuracy']==models['Test Accuracy'].max()].head(n=5)
best_index = models[models['Test Accuracy']==models['Test Accuracy'].max()]['Test Precision'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
(models.iloc[best_index, :])

m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))
success=m_confusion_test[1,1]+m_confusion_test[0,0]
total_n=success+m_confusion_test[1,0]+m_confusion_test[0,1]
print(success/total_n)