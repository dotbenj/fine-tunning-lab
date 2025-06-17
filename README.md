# 🚀 Fine-Tuning Lab — Classification de Profils LinkedIn

Bienvenue dans ce projet pédagogique de fine-tuning !  
L’objectif est de classifier des expériences professionnelles en départements (marketing, finance, tech, etc.) à partir de données issues de profils LinkedIn.

Ce dépôt contient **plusieurs versions du même projet**, adaptées à différents environnements matériels :

- ✅ `cpu_version/` : pour machines sans GPU ou avec faible puissance
- ⚡ `gpu_version/` : optimisée pour PC avec carte Nvidia (ex: série 30/40)
- 🍎 `apple_silicon_version/` : pour Mac M1, M2 ou M3 (via backend MPS)

---

## 🧠 Objectif du projet

**Tâche** : classification supervisée (`text → label`)  
**Entrée** : un `title` + une `description` d’un poste  
**Sortie** : un label de département, ex : `tech`, `sales`, `legal`, etc.

Exemple de ligne dans le jeu de données :
```json
{
  "title": "Responsable Marketing Digital",
  "description": "Développement de la stratégie d'acquisition online.",
  "label": "marketing"
}
```

## 🛠️ Instructions générales

1 - Clonez le repo :

```bash
git clone https://github.com/<votre-utilisateur>/fine-tuning-lab.git
cd fine-tuning-lab
```

2 - Installez les dépendances dans l’un des répertoires :

```bash
cd cpu_version
pyhton -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3 - Lancer l'entrainementt :

```bash
python train.py
```
