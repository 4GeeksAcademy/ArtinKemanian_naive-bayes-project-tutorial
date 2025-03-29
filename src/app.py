from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from pickle import dump

datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv")

datos = datos.drop_duplicates().reset_index(drop = True)
datos = datos.drop("package_name", axis = 1)

datos["review"] = datos["review"].str.strip().str.lower()

X = datos["review"]
y = datos["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()

modelado = MultinomialNB()
modelado.fit(X_train, y_train)

y_pred = modelado.predict(X_test)

accuracy_score(y_test, y_pred)

for modelado_aux in [GaussianNB(), BernoulliNB()]:
    modelado_aux.fit(X_train, y_train)
    y_pred_aux = modelado_aux.predict(X_test)
    print(f"{modelado_aux} with accuracy: {accuracy_score(y_test, y_pred_aux)}")

hiperparametros = {
    "alpha": np.linspace(0.01, 10.0, 200),
    "fit_prior": [True, False]
}

random_search = RandomizedSearchCV(modelado, hiperparametros, n_iter = 50, scoring = "accuracy", cv = 5, random_state = 42)
random_search

random_search.fit(X_train, y_train)

print(f"Mejores hiperparámetros: {random_search.best_params_}")

modelado = MultinomialNB(alpha = 1.917638190954774, fit_prior = False)
modelado.fit(X_train, y_train)
modelado.fit(X_train, y_train)
y_pred = modelado.predict(X_test)

accuracy_score(y_test, y_pred)

dump(modelado, open("models/naive_bayes_alpha", "wb"))