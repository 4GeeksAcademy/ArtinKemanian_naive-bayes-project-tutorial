# Naive Bayes - Guía paso a paso

- Comprender un dataset nuevo.
- Procesarlo aplicando un análisis exploratorio (EDA).
- Modelar los datos utilizando un Naive Bayes.
- Analizar los resultados y optimizar el modelo.

## 🌱 Cómo iniciar este proyecto

Sigue las siguientes instrucciones:

1. Crea un nuevo repositorio basado en el [proyecto de Machine Learning](https://github.com/4GeeksAcademy/machine-learning-python-template) [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Abre el repositorio creado recientemente en Codespace usando la [extensión del botón de Codespace](https://docs.github.com/es/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Una vez que el VSCode del Codespace haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

### Análisis de sentimientos

Los modelos Naive Bayes son muy útiles cuando queremos analizar sentimientos, clasificar textos en tópicos o recomendaciones, ya que las características de estos desafíos cumplen muy bien con los supuestos teóricos y metodológicos del modelo.

En este proyecto practicarás con un conjunto de datos para crear un clasificador de reseñas de la tienda de Google Play.

#### Paso 1: Carga del conjunto de datos

El conjunto de datos se puede encontrar en esta carpeta de proyecto bajo el nombre `playstore_reviews.csv`. Puedes cargarlo en el código directamente desde el sigiente enlace: 

```text
https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv
```
O descargarlo y añadirlo a mano en tu repositorio. En este conjunto de datos encontrarás las siguientes variables:

- `package_name`. Nombre de la aplicación móvil (categórico)
- `review`. Comentario sobre la aplicación móvil (categórico)
- `polarity`. Variable de clase (0 o 1), siendo 0 un comentario negativo y 1, positivo (categórico numérico)

#### Paso 2: Estudio de variables y su contenido

En este caso, tenemos solo 3 variables: 2 predictoras y una etiqueta dicotómica. De las dos predictoras, realmente solo nos interesa la parte del comentario, ya que el hecho de clasificar un comentario en positivo o negativo dependerá de su contenido, no de la aplicación de la que se haya escrito. Por lo tanto, la variable `package_name` habría que eliminarla.

Cuando trabajamos con textos como en este caso, no tiene sentido hacer un EDA, el proceso es diferente, ya que la única variable que nos interesa es la que contiene el texto. En otros casos en los que el texto formase parte de un conjunto complejo con otras variables predictoras numéricas y el objetivo de predicción sea distinto, entonces tiene sentido aplicar un EDA.

Sin embargo, no podemos trabajar con texto plano, antes hay que procesarlo. Este proceso consta de varios pasos:

1. Eliminar espacios y convertir a minúsculas el texto:
```py
df["column"] = df["column"].str.strip().str.lower()
```
2. Dividir el conjunto de datos en train y test: `X_train`, `X_test`, `y_train`, `y_test`
3. Transformar el texto en una matriz de recuento de palabras. Esta es una forma de obtener características numéricas a partir del texto. Para ello, utilizamos el conjunto de train para entrenar el transformador y la aplicamos en test:
```py
vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()
```

Una vez hayamos terminado tendremos listas las predictoras para entrenar el modelo.

#### Paso 3: Construye un naive bayes

Comienza a resolver el problema implementando un modelo del que tendrás que elegir cuál de las tres implementaciones utilizar: `GaussianNB`, `MultinomialNB` o `BernoulliNB`, según lo que hemos estudiado en el módulo. Prueba ahora a entrenarlo con las dos otras implementaciones y confirma si el modelo que has elegido es el adecuado.

#### Paso 4: Optimiza el modelo anterior

Después de entrenar el modelo en sus tres implementaciones, elige la mejor opción y trata de optimizar sus resultados con un random forest, si es posible.

#### Paso 5: Guarda el modelo

Almacena el modelo en la carpeta correspondiente.

#### Paso 6: Explora otras alternativas

¿Qué otros modelos de los que hemos estudiado podrías utilizar para intentar superar los resultados de un Naive Bayes? Arguméntalo y entrena el modelo.

> Nota: También incorporamos muestras de solución en `./solution.ipynb` que te sugerimos honestamente que solo uses si estás atascado por más de 30 minutos o si ya has terminado y quieres compararlo con tu enfoque.