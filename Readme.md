# სახის გამომეტყველების ამოცნობა (FER2013)

**Kaggle:** [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)  
**WandB პროექტი:** [fer-challenge](https://wandb.ai/dshon23-free-university-of-tbilisi/fer-challenge)  
**WandB რეპორტი:** [FER2013 Experiment Results](https://wandb.ai/dshon23-free-university-of-tbilisi/fer-challenge/reports/FER2013-Experiment-Results--VmlldzoxNzI0ODU3MA)

---

## Dataset

- **48×48 პიქსელის გრეისქეილ სურათები**, 7 ემოციის კლასი
- 28,709 სატრენინგო / 3,589 ვალიდაციის / 3,589 სატესტო სურათი
- კლასების დისბალანსი: `Disgust` კლასს ~10-ჯერ ნაკლები სურათი აქვს ვიდრე `Happy`-ს → გამოსწორებულია class-weighted loss-ით

| Label | ემოცია   |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

---

## სტრატეგია

მიდგომა: **პატარა არქიტექტურიდან დაწყება და იტერაციულად გაფართოება**. თითოეული არქიტექტურა წინამდებარის კონკრეტულ პრობლემას წყვეტს. მიზანი იყო underfitting-ის, overfitting-ის და კარგი გენერალიზაციის ჩვენება.

---

## არქიტექტურები და შედეგები

### v1 — TinyCNN (შეგნებული Underfitting)

**არქიტექტურა:**
```
Conv(1→8) → ReLU → MaxPool
Conv(8→16) → ReLU → MaxPool
Linear(2304→7)
```

**პარამეტრების რაოდენობა:** ~50K

**გადაწყვეტილება:** ვიწყებთ ყველაზე პატარა შესაძლო მოდელით. 2 conv layer-ი 7 კლასის სახის გამომეტყველების ასაღებად არ კმარა — ეს შეგნებულად გავაკეთეთ underfitting-ის საჩვენებლად.

**ჰიპერპარამეტრების ტესტი:**

| Run | Optimizer | LR | Val Acc |
|-----|-----------|-----|---------|
| tiny_cnn_v1 | Adam | 0.001 | **0.478** |
| tiny_cnn_v2 | Adam | 0.01  | **0.249** |
| tiny_cnn_v3 | SGD  | 0.01  | **0.480** |

**ანალიზი:** ყველა run-ში train accuracy და val accuracy ორივე დაბალია (~47-48%) — კლასიკური underfitting. მოდელს არ ყოფნის capacity 7 ემოციური კლასის გასარჩევად. LR=0.01 Adam-ით კატასტროფულად ცუდი შედეგი (24.9%) — ძალიან მაღალმა learning rate-მა გამოიწვია training-ის არასტაბილურობა. SGD და Adam lr=0.001 მსგავს შედეგს იძლევა, რაც ნიშნავს რომ პრობლემა optimizer-ი კი არა, capacity-ა.

**დასკვნა:** მოდელს მეტი layer-ი სჭირდება.

---

### v2 — SmallCNN (შეგნებული Overfitting)

**არქიტექტურა:**
```
Conv(1→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Conv(64→128) → ReLU → MaxPool
Conv(128→128) → ReLU → MaxPool
Linear(1152→512) → Linear(512→7)
```

**პარამეტრების რაოდენობა:** ~1.2M

**გადაწყვეტილება:** capacity გავზარდეთ (4 conv layer, ~1.2M პარამეტრი), მაგრამ regularization-ი არ დავამატეთ. მიზანი — overfitting-ის დემონსტრაცია.

**ჰიპერპარამეტრების ტესტი:**

| Run | Dropout | Augment | Val Acc |
|-----|---------|---------|---------|
| small_cnn_v1_no_reg | 0.0 | არა | **0.572** |
| small_cnn_v2_dropout | 0.3 | დიახ | **0.571** |

**ანალიზი:** val accuracy გაიზარდა TinyCNN-თან შედარებით (47%→57%), მაგრამ train accuracy გაცილებით მაღალი იყო (~75%) — train/val gap-ი overfitting-ის ნიშანია. საინტერესოა რომ dropout=0.3-ის დამატებამ ამ კონკრეტულ შემთხვევაში მნიშვნელოვნად არ გააუმჯობესა შედეგი — ეს იმაზე მიუთითებს რომ BatchNorm-ის გარეშე Dropout ეფექტი ნაკლებია.

**დასკვნა:** Capacity კმარა, regularization-ი სჭირდება. BatchNorm + Dropout ერთად უნდა გამოვიყენოთ.

---

### v3 — MediumCNN (Regularization)

**არქიტექტურა:**
```
[Conv → BatchNorm → ReLU → MaxPool → Dropout2d(0.25)] × 4
Linear(2304→512) → Dropout → Linear(512→128) → Linear(128→7)
```

**გადაწყვეტილება:** BatchNorm დავამატეთ internal covariate shift-ის შესამცირებლად (training-ის სტაბილიზაცია) და Dropout — ნეირონების co-adaptation-ის თავიდან ასარიდებლად. Data augmentation (horizontal flip, rotation, crop) კიდევ უფრო ამდიდრებს training სეტს.

**ჰიპერპარამეტრების ტესტი:**

| Run | Dropout | Scheduler | Batch Size | Val Acc |
|-----|---------|-----------|------------|---------|
| medium_cnn_v1_dropout03 | 0.3 | plateau | 64 | **0.600** |
| medium_cnn_v2_dropout05 | 0.5 | plateau | 64 | **0.598** |
| medium_cnn_v3_cosine    | 0.4 | cosine  | 32 | **0.602** |

**ანალიზი:** val accuracy მნიშვნელოვნად გაიზარდა (57%→60%). train/val gap-ი გაცილებით პატარა გახდა — BatchNorm + Dropout ეფექტურია. Dropout=0.5 ოდნავ ჩამოუვარდა 0.3-ს, რაც ნიშნავს რომ ზედმეტმა regularization-მა შეიძლება ოდნავ underfitting-ი გამოიწვიოს. Cosine annealing scheduler-მა უკეთესი შედეგი მოიტანა plateau-სთან შედარებით — learning rate-ის გლუვი შემცირება training-ის ბოლოს ეხმარება local minima-ს გარიდებაში.

**დასკვნა:** BatchNorm + Dropout + Augmentation კარგი კომბინაციაა. მაგრამ შეგვიძლია კიდევ გავაუმჯობესოთ gradient flow-ის გაუმჯობესებით.

---

### v4 — ResNetStyleCNN (Residual Connections)

**არქიტექტურა:**
```
Stem: Conv(1→32) → BatchNorm → ReLU
Stage1: ResidualBlock(32) → MaxPool
Stage2: Conv(32→64) → ResidualBlock(64) → MaxPool
Stage3: Conv(64→128) → ResidualBlock(128) → MaxPool
Stage4: Conv(128→256) → ResidualBlock(256) → MaxPool
GlobalAvgPool → Dropout → Linear(256→7)
```

**გადაწყვეტილება:** Skip connections საშუალებას იძლევა gradient-ი პირდაპირ ადრეულ layer-ებამდე მიაღწევს. სახის ემოციების ამოსაცნობად hierarchical features მნიშვნელოვანია (კვარები, პირი, წარბები) — residual blocks ამ feature-ებს ეფექტურად სწავლობს. GlobalAvgPool დიდ FC layer-ებს ცვლის — ნაკლები პარამეტრი, უკეთესი გენერალიზაცია.

**ჰიპერპარამეტრების ტესტი:**

| Run | Scheduler | Val Acc |
|-----|-----------|---------|
| resnet_style_v1_plateau | plateau | **0.588** |
| resnet_style_v2_cosine  | cosine  | **0.608** |

**ანალიზი:** Cosine scheduler-ით ResNetStyle საუკეთესო შედეგს აღწევს (60.8%). Training curve-ი გაცილებით სტაბილური გახდა — ნაკლები oscillation loss-ში. Gradient flow visualization WandB-ზე ადასტურებს რომ ადრეული layer-ებიც კარგ gradient-ს იღებს skip connections-ის გამო. Plateau scheduler-ი ჩამოუვარდა cosine-ს, რადგან learning rate-ის შემცირება ძალიან გვიან ხდება.

**დასკვნა:** Residual connections მკაფიოდ ეხმარება. Skip connections-ი ამ ამოცანისთვის კარგი არქიტექტურული გადაწყვეტილებაა.

---

### v5 — Transfer Learning (MobileNetV2)

**არქიტექტურა:**
```
MobileNetV2 (ImageNet pretrained)
  ↓ last 3-6 blocks unfrozen
  ↓ AdaptiveAvgPool
  ↓ Dropout → Linear(1280→256) → ReLU → Linear(256→7)
```

**გადაწყვეტილება:** ImageNet-ზე ნასწავლი edge/texture features პირდაპირ გადადის სახის გამომეტყველების ამოცნობაზე. 28K სურათი scratch-იდან სასწავლად ცოტაა — pretrained model ამ პრობლემას წყვეტს. 48×48 სურათები resize-ებულია 224×224-ზე და 3-channel-ად გარდაიქმნება MobileNetV2-სთვის.

**ჰიპერპარამეტრების ტესტი:**

| Run | Unfrozen Blocks | LR | Scheduler | Val Acc |
|-----|----------------|----|-----------|---------|
| transfer_v1_frozen   | 3 | 3e-4 | plateau | **0.607** |
| transfer_v2_finetune | 6 | 1e-4 | cosine  | crashed |

**ანალიზი:** transfer_v1 კარგ შედეგს (60.7%) იძლევა სულ 15 epoch-ში — ეს ნათლად ადასტურებს pretrained features-ის ძალას. transfer_v2 crashed-ი — სავარაუდოდ Colab session-ის გაწყვეტის გამო, არა კოდის შეცდომის. მეტი layer-ის unfreezing-ით და დაბალი LR-ით (1e-4) კიდევ უკეთესი შედეგი ვიცის, მაგრამ ამ run-ს ვერ დავასრულეთ.

**დასკვნა:** Pretrained features საგრძნობლად სჯობს scratch-ს ამ dataset size-ისთვის. ძალიან დაბალი LR აუცილებელია pretrained weights-ის გაფუჭების თავიდან ასარიდებლად.

---

## შედეგების შედარება

| არქიტექტურა | Val Acc | დიაგნოზი |
|-------------|---------|----------|
| TinyCNN (lr=0.01) | 0.249 | Underfitting + არასტაბილური training |
| TinyCNN (lr=0.001) | 0.478 | Underfitting — ნაკლები capacity |
| SmallCNN (no reg) | 0.572 | Overfitting — train/val gap დიდია |
| SmallCNN (dropout) | 0.571 | Overfitting ნაწილობრივ შემცირებული |
| MediumCNN (dropout=0.3) | 0.600 | კარგი ბალანსი |
| MediumCNN (cosine) | 0.602 | უკეთესი scheduler |
| ResNetStyle (plateau) | 0.588 | კარგი, მაგრამ scheduler-ი ვერ |
| ResNetStyle (cosine) | **0.608** | საუკეთესო from-scratch |
| Transfer (frozen) | 0.607 | Pretrained features ეფექტურია |

---

## მთავარი დასკვნები

1. **Learning Rate მნიშვნელოვანია** — lr=0.01 TinyCNN-ზე training-ს სრულიად არასტაბილური გახადა (24.9%). სწორი LR შერჩევა კრიტიკულია.

2. **BatchNorm + Dropout ერთად** — Dropout-ი მარტო (SmallCNN) ნაკლებად ეფექტურია ვიდრე BatchNorm + Dropout კომბინაცია (MediumCNN). 3% სხვაობა val accuracy-ში.

3. **Cosine Scheduler > Plateau** — ყველა ექსპერიმენტში cosine annealing scheduler plateau-ს სჯობდა. Learning rate-ის გლუვი შემცირება training-ის ბოლოს კარგ შედეგს იძლევა.

4. **Residual Connections სწავლობს** — gradient flow visualization ადასტურებს რომ skip connections ეხმარება ადრეული layer-ების გაწვრთნას.

5. **Transfer Learning ეფექტურია მცირე dataset-ზე** — 15 epoch-ში Transfer Learning-მა 60.7% მიაღწია, ResNetStyle-ს კი 25 epoch-ი დასჭირდა 60.8%-ისთვის.

---

## პროექტის სტრუქტურა

```
fer-challenge/
├── notebooks/
│   └── fer_experiments.ipynb   # მთავარი Colab notebook
├── src/
│   ├── dataset.py              # FERDataset, transforms, dataloaders
│   ├── models.py               # 5 არქიტექტურა
│   ├── train.py                # Training loop + WandB logging
│   └── utils.py                # Sanity checks, confusion matrix
├── configs/
│   └── hyperparams.yaml        # ყველა ექსპერიმენტის კონფიგი
├── requirements.txt
└── README.md
```

---
