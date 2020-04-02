
# API de transfert de Style

### *Par Benjamin Nicol*

Cette API permet d'utiliser une technologie du machine learning qui est le transfert de style. Elle utilise Tensorflow  Lite ainsi qu'un modèle adapté pour une utilisation sur environnement embarqué.

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
L'API dispose de trois routes pour le moment :

- GET :
	- / : Renvoie des informations relatives à l'API
	- /example : Permet d'avoir un exemple de ce que peux renvoyer l'API

- POST :
	- /model/**\<modelName\>** : A utiliser avec une image dans un attribut **image**, renvoit l'image envoyée avec le style donnée dans le paramètre **\<modelName\>**

Exemples d'utilisation :
```
# Renvoie le résultat de l'exemple
$ curl -o example.jpg http://localhost:5000/example

# Renvoie le fichier 'lance.jpg' avec le style 'sphere' dans le fichier 'output.jpg'
$ curl --request POST 'http://127.0.0.1:5000/model/udnie' --form 'file=@/home/bnicol/Pictures/lance.jpg' --output 'output.jpg'
```
## Structure de fichiers
L'API est constiuée du dossier racine contenant les fichiers nécessaires au fonctionnement de celle-ci et de trois sous-dossiers 

- styles : contenant les différents styles disponibles
- upload : ici sont sauvegardées les images envoyées par le POST avant le traitement
- save : ici sont sauvegardées les images après le transfert de style

## Styles
| Nom du style (\<modelName\>) | Tableau original |
|--|--|
| udnie | Udnie, par Francis Picabia![Udnie, Young American Girl by Francis Picabia](https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Francis_Picabia%2C_1913%2C_Udnie_%28Young_American_Girl%2C_The_Dance%29%2C_oil_on_canvas%2C_290_x_300_cm%2C_Mus%C3%A9e_National_d%E2%80%99Art_Moderne%2C_Centre_Georges_Pompidou%2C_Paris..jpg/599px-Francis_Picabia%2C_1913%2C_Udnie_%28Young_American_Girl%2C_The_Dance%29%2C_oil_on_canvas%2C_290_x_300_cm%2C_Mus%C3%A9e_National_d%E2%80%99Art_Moderne%2C_Centre_Georges_Pompidou%2C_Paris..jpg) |
| scream | Le cri, par Edvard Munch<br/>![Le cri, par Edvard Munch](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/260px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg)|
| sphere | Hand with Reflecting Sphere, par M.C. Escher<br/>![Hand with Reflecting Sphere, par M.C. Escher](https://upload.wikimedia.org/wikipedia/en/6/66/Hand_with_Reflecting_Sphere.jpg)|

