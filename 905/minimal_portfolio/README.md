## Portfolio — Mode d’emploi (simple)

Un petit site vitrine, facile à mettre à jour.

### Ouvrir le portfolio sur votre ordinateur

1) Ouvrez le dossier « PORTFOLIO copy ».
2) Ouvrez un Terminal dans ce dossier.
3) Lancez le site avec cette commande:

```
python3 -m http.server 8000
```

4) Dans votre navigateur, allez à:

`http://127.0.0.1:8000/sources/gaydar_algorithm.html`

Astuce: n’ouvrez pas les fichiers en double-cliquant (cela casse la bannière). Passez toujours par l’adresse qui commence par `http://` (`http://127.0.0.1:8000/sources/gaydar_algorithm.html` par exemple).

### Créer une nouvelle page

1) Dupliquez `sources/gaydar_algorithm.html` et renommez le fichier (par ex. `sources/mon_projet.html`).
2) Ouvrez votre nouveau fichier et:
- changez le texte du `<title>` (onglet du navigateur),
- changez le grand titre `<h1>`,
- remplacez l’image dans la section visuel (mettez la vôtre dans `medias/` et pointez vers `../medias/mon_image.png`).
3) Laissez la ligne suivante telle quelle (elle affiche automatiquement la bannière commune):

`<div data-include="nav"></div>`

### Ajouter votre page dans la bannière (menu du haut)

1) Ouvrez `sources/nav.html`.
2) Copiez une ligne de menu existante (elle commence par `<li><a …`), collez-la en dessous, puis:
- remplacez le nom du fichier (ex. `mon_projet.html`),
- remplacez le texte visible (le nom de votre projet).
3) Détail important: dans le lien, la partie `data-page="..."` doit être exactement le nom du fichier sans `.html` (ex. `data-page="mon_projet"`). Cela permet de surligner automatiquement la page active.

### Ajouter des images et des vidéos

- Mettez tous vos fichiers médias dans le dossier `medias/`.
- Dans vos pages (qui sont dans `sources/`), référencez-les avec `../medias/nom_du_fichier.ext`.

### Mettre en ligne

- Envoyez le dossier entier tel quel chez votre hébergeur (il doit contenir `sources/` et `medias/`).
- L’adresse d’une page ressemblera à: `https://votresite/sources/mon_projet.html`.

### Si quelque chose ne s’affiche pas

- La bannière a disparu: vérifiez que vous passez par une adresse en `http://` (serveur lancé) et pas par un fichier ouvert directement.
- Le lien du menu n’est pas en surbrillance: vérifiez `data-page` dans `sources/nav.html` (il doit correspondre au nom du fichier sans `.html`).
- Une image ne s’affiche pas: vérifiez que l’image est bien dans `medias/` et que le chemin commence par `../medias/`.

### Où changer l’apparence et le comportement

- Apparence (couleurs, tailles, etc.): `sources/style.css`
- Comportements simples (affichage de la bannière, colonnes de texte): `sources/common.js`



