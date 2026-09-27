\# AgriShield-AI — Machine Learning Results



\## 1. Dataset



\- Dataset: PlantVillage

\- Crops: Corn, Grape, Potato, Tomato

\- Number of classes: 21

\- Total images: 28,225

\- Valid images: 28,225

\- Corrupt images: 0



\## 2. Dataset Split



The dataset was divided into:



\- Training: 70%

\- Validation: 15%

\- Testing: 15%



\### Number of Images



\- Training: 19,751

\- Validation: 4,233

\- Testing: 4,241



\## 3. Model



The project uses an ImageNet-pretrained EfficientNetB0 model.



\### Training Approach



1\. Transfer learning with the EfficientNetB0 base initially frozen.

2\. Custom classification head for 21 classes.

3\. Fine-tuning of the pretrained model using a low learning rate.

4\. Class weights were used to address class imbalance.



\## 4. Final Test Results



The fine-tuned EfficientNetB0 model achieved:



| Metric | Score |

|---|---:|

| Accuracy | 92.57% |

| Macro Precision | 91.71% |

| Macro Recall | 93.31% |

| Macro F1-Score | 92.29% |

| Weighted Precision | 93.17% |

| Weighted Recall | 92.57% |

| Weighted F1-Score | 92.70% |



\## 5. Test-Time Augmentation



Horizontal-flip Test-Time Augmentation (TTA) was evaluated.



| Configuration | Accuracy |

|---|---:|

| Normal prediction | 92.57% |

| Horizontal Flip TTA | 92.74% |

| Multi-TTA | 91.79% |



Horizontal-flip TTA improved the measured test accuracy by 0.17 percentage points.



The multi-TTA configuration using additional vertical flip and 90-degree rotation reduced accuracy and was therefore not selected.



\## 6. Evaluation Files



The following evaluation files are included in the repository:



\- `classification\_report.txt`

\- `confusion\_matrix.png`

\- `tta\_results.txt`

\- `multi\_tta\_results.txt`



\## 7. Model File



The trained model is:



`efficientnetb0\_finetuned.keras`



The model was trained using the 21 selected PlantVillage classes.



\## 8. Prediction



The `predict.py` script can load the trained model and perform disease classification on a new leaf image.



Example:



```powershell

python scripts/predict.py path\\to\\leaf\_image.jpg

