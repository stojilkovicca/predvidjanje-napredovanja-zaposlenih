# Predviđanje napredovanja zaposlenih — komparativna analiza klasifikacionih modela

Diplomski rad · Fakultet organizacionih nauka, Univerzitet u Beogradu

Poređenje tri algoritma mašinskog učenja na problemu predviđanja unapređenja zaposlenih,
na izrazito nebalansiranom skupu podataka (9,88 % pozitivne klase). Svaki algoritam je
posmatran u tri varijante, čime se doprinos samog algoritma razdvaja od doprinosa
načina na koji je pripremljen.

---

## Postavka

| | |
|---|---|
| **Zadatak** | binarna klasifikacija |
| **Skup podataka** | [Employee Promtion Prediction Dataset](https://www.kaggle.com/datasets/rohit8527kmr7518/employee-promtion-prediction) (Kaggle, CC0, sintetički) |
| **Obim** | 100.000 zapisa × 43 kolone → 92.458 posle čišćenja |
| **Disbalans klasa** | 9,88 % unapređenih (1 : 9,1) |
| **Podela** | 80 / 20, stratifikovano, `random_state=42` |
| **Validacija** | `StratifiedKFold`, pet preklopa |
| **Glavna mera** | F1 (tačnost obmanjuje pri ovakvom disbalansu) |

Shema 3 × 3 — tri algoritma, svaki u tri varijante:

| | osnovni | balansirani | podešeni |
|---|---|---|---|
| **Logistička regresija** | podrazumevani parametri | `class_weight='balanced'` | `GridSearchCV` |
| **Random Forest** | podrazumevani parametri | `class_weight='balanced_subsample'` | `GridSearchCV` |
| **XGBoost** | podrazumevani parametri | `scale_pos_weight` | `GridSearchCV` |

---

## Rezultati

Na test skupu (18.492 zapisa, 1.827 unapređenih), pri pragu 0,5:

| Model | Accuracy | Precision | Recall | **F1** | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LR osnovni | 0,9120 | 0,6448 | 0,2425 | 0,3524 | 0,8658 | 0,4703 |
| LR balansirani | 0,7766 | 0,2784 | 0,7920 | 0,4120 | 0,8656 | 0,4700 |
| LR podešeni | 0,7765 | 0,2783 | 0,7920 | 0,4118 | 0,8656 | 0,4701 |
| RF osnovni | 0,9055 | 0,8264 | 0,0547 | 0,1027 | 0,8688 | 0,4720 |
| RF balansirani | 0,9059 | 0,8480 | 0,0580 | 0,1086 | 0,8846 | 0,5083 |
| RF podešeni | 0,8863 | 0,4417 | 0,5720 | 0,4984 | 0,8868 | 0,5190 |
| XGB osnovni | 0,9227 | 0,6764 | 0,4176 | 0,5164 | 0,9275 | 0,6208 |
| XGB balansirani | 0,8863 | 0,4543 | 0,7515 | 0,5663 | 0,9256 | 0,6220 |
| **XGB podešeni** | 0,9028 | 0,5058 | 0,6935 | **0,5849** | **0,9292** | **0,6421** |

Referentna tačka: trivijalan model koji sve proglašava unapređenima daje F1 = 0,1799.

**Nalazi:**

- XGBoost nadmašuje ostala dva algoritma u svim varijantama;
- balansiranje klasa dramatično menja Random Forest (F1 sa 0,10 na 0,50 uz podešavanje),
  dok kod logističke regresije podiže odziv uz pad preciznosti;
- optimizacija praga donosi najviše slabijim modelima — RF osnovni skače sa 0,1027 na 0,4743,
  dok kod već podešenog XGBoost-a dobitak iznosi svega 0,0056;
- 33 od 36 poređenja među modelima statistički su značajna (uparen bootstrap, 1000 uzoraka).

---

## Struktura

```
analiza.ipynb      celokupna analiza, 100 ćelija (A–F)
pretraga_rf.py     zasebna pretraga hiperparametara za Random Forest
slike/             13 grafikona (200 dpi)
tabele/            17 tabela sa rezultatima (CSV)
requirements.txt   verzije korišćenih biblioteka
```

Beležnica je podeljena u faze: **A** priprema i čišćenje · **B** eksplorativna analiza ·
**C** priprema za modelovanje · **D** kreiranje modela · **E** evaluacija ·
**F** prag klasifikacije i značajnost atributa.

---

## Pokretanje

```bash
pip install -r requirements.txt
```

Skup podataka nije u repozitorijumu zbog veličine. Preuzima se sa
[Kaggle stranice](https://www.kaggle.com/datasets/rohit8527kmr7518/employee-promtion-prediction)
i smešta u koren projekta kao `employee_promotion_prediction.csv`.

Zatim se beležnica pokreće redom, od prve do poslednje ćelije. Pretraga hiperparametara
za Random Forest traje oko sat vremena; ako je već pokrenuta preko `pretraga_rf.py`,
beležnica automatski učitava sačuvani rezultat.

Svi koraci koji sadrže slučajnost koriste `random_state=42`, pa se dobijaju identični brojevi.

---

## Metodološke napomene

- **VIF eliminacija** je sprovedena isključivo na trening skupu, radi izbegavanja curenja podataka;
  linearni i stablasti modeli zato koriste različite skupove atributa.
- **Prag klasifikacije** biran je pomoću `cross_val_predict` na trening skupu, ne na testu.
- **McNemarov test** poredi modele po tačnosti, pa je kao dopuna uveden **bootstrap interval
  poverenja za razliku u F1** — meri koja nosi zaključak rada. Uzorci su upareni.
- **Grupna permutaciona važnost** meša ceo blok korelisanih atributa odjednom, jer pojedinačna
  važnost potcenjuje blokove u kojima se atributi međusobno zamenjuju.

---

## Ograničenja

Skup podataka je **sintetički** i sam izvor navodi da sadrži skrivene činioce koji utiču na
unapređenje, a nisu uključeni u podatke. To postavlja gornju granicu dostižnoj tačnosti i
znači da se zaključci odnose na ovaj skup, a ne nužno na stvarno organizaciono okruženje.

---

## Licenca

Kod je objavljen pod [MIT licencom](LICENSE). Skup podataka je vlasništvo autora sa Kaggle-a
i objavljen je pod CC0.
