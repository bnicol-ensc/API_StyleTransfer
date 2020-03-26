# API de transfert de Style
### *Par Benjamin Nicol*
Cette API permet d'utiliser une technologie du machine learning qui est le transfert de style.

Cette API utilise les modèles de transfert de style publiés par Google.

## Installation
Afin que l'API fonctionne correctement, il suffit d'installer les deux dépendances suivantes : TensorFlow et Flask.
```
$ pip install Tensorflow
$ pip install Flask
```
Et de la lancer en utilisant python (3.7)
```
$ python api.py
```
## Routes
L'API dispose de trois route pour le moment :
- GET :
	- / : Route d'accueil (ne sert à rien pour le moment)
	 - /example : Permet d'avoir un exemple de ce que peux renvoyer l'API
 - POST :
	 - /testInput : A utiliser avec une image dans un attribut **image**, renvoit l'image envoyée avec le style de l'exemple

## Styles
Pour le moment seul un style est disponible. L'ajout de nouveaux styles sera réalisé dans un prochaine version.

Style utilisé :  Udnie, by Francis Picabia (1879-1953)
![Udnie, Young American Girl by Francis Picabia](https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Francis_Picabia%2C_1913%2C_Udnie_%28Young_American_Girl%2C_The_Dance%29%2C_oil_on_canvas%2C_290_x_300_cm%2C_Mus%C3%A9e_National_d%E2%80%99Art_Moderne%2C_Centre_Georges_Pompidou%2C_Paris..jpg/599px-Francis_Picabia%2C_1913%2C_Udnie_%28Young_American_Girl%2C_The_Dance%29%2C_oil_on_canvas%2C_290_x_300_cm%2C_Mus%C3%A9e_National_d%E2%80%99Art_Moderne%2C_Centre_Georges_Pompidou%2C_Paris..jpg)

Cela donne les résultats suivant sur l'exemple :
![Résultat de l'exemple](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/formula.png)
