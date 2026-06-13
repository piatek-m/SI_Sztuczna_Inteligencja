# Projekt - torchvision, klasyfikacja owocy i warzyw
  
### Dataset
Wykorzystano dataset _Food and Vegetables_ ([huggingface](https://huggingface.co/datasets/SunnyAgarwal4274/Food_and_Vegetables)) użytkownika _SunnyAgarwal4274_.

Dataset miał kilka problemów utrudniających uczenie, generalizację:
1. Semantyka klas - 2 przypadki klas, które określały ten sam obiekt - rozwiązano poprzez scalenie klas:
   - ```bell pepper <-> capsicum```
   - ```corn <-> sweetcorn```
2. Duplikaty między klasami - niektóre pary klas miały identyczne obrazy np. ```bell pepper_19.jpg <-> capsicum_2.jpg```. Rozwiązano poprzez usunięcie jednego z obrazów po mergu, zob. pkt 3. i 4.
3. Data leakage - duplikaty obrazów w ```train/``` oraz ```val/```. Rozwiązano poprzez porównanie hashy MD5 oraz usunięcie duplikatu z ```train/```.
4. Duplikaty w obrębie jednej klasy **i** jednego zestawu np. ```train/bell pepper_26.jpg <-> train/bel pepper_67.jpg```. Usunięto jeden z duplikatów.

Katalog ```train``` splitowany jest w ```src/train.py``` na ```train_subset``` i ```val_subset```, natomiast katalog ```val``` używany jedynie w ```src/eval.py```.
Oryginalny dataset miał **36** klas i **3114** obrazów treningowych. Po deduplikacji ma **34** klasy, **2585** obrazów treningowych (całość ```train```, przed podziałem na podzbiory). 

Wzbogacono dane (```src/data/transforms.py```) poprzez np. ```RandomResizedCrop```, ```RandomVerticalFlip```, ```ColorJitter```.

### Model i uczenie

Wykorzystano uczenie transferowe. Jako podstawy użyto [_MobileNetV3-Large_](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html). Uczono w dwóch fazach:
1. Głowica
2. Finetuning szkieletu (backbone)

Zastosowano scheduler learning rate ```ReduceLROnPlateau```. Zrezygnowano z CosineannealingLR z powodu zbyt małego datasetu.

---

## Zadanie KNN, DT, RF - użyte datasety:
> Linki są także zakomentowane w plikach .py w pierwszej linii.
- [KNN](https://huggingface.co/datasets/mstz/heart_failure)
- [DecisionTree](https://huggingface.co/datasets/scikit-learn/auto-mpg)
- [RandomForest](https://huggingface.co/datasets/wwydmanski/wisconsin-breast-cancer)
