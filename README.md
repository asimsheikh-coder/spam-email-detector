# 📧 Spam Email Detector using Naive Bayes

## 📄 Abstract
This project demonstrates how to classify emails as **spam** or **ham (not spam)** using the **Naive Bayes** algorithm.  
It is a beginner-friendly project that shows the core ideas of **Natural Language Processing (NLP)** and **Machine Learning** using simple Python libraries.  
The notebook processes text data using **TF-IDF** and applies the **Multinomial Naive Bayes** classifier to detect spam emails effectively.

---

## 🎯 Objectives
1. Learn how Naive Bayes works for text classification.  
2. Preprocess text data using TF-IDF (Term Frequency–Inverse Document Frequency).  
3. Build a spam classifier using Scikit-learn.  
4. Evaluate model accuracy and visualize confusion matrix.  
5. Test the model on unseen text samples.

---

## 🧩 Dataset
A small demo dataset is included directly inside the notebook for simplicity.  
It contains 10 example messages labeled as “spam” or “ham.”  

You can replace it with the **[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)** for real-world training.

---

## ⚙️ Implementation Steps
| Step | Description |
|------|--------------|
| 1️⃣ | **Import Libraries** — Load scikit-learn, pandas, and visualization tools. |
| 2️⃣ | **Load Dataset** — Create or import a labeled dataset of messages. |
| 3️⃣ | **Preprocess Text** — Convert messages into numerical features using TF-IDF. |
| 4️⃣ | **Train Model** — Apply Multinomial Naive Bayes to learn patterns. |
| 5️⃣ | **Evaluate Model** — Check accuracy, confusion matrix, and classification report. |
| 6️⃣ | **Predict New Messages** — Test the model on new unseen inputs. |

---

## 🧠 Understanding Naive Bayes
The Naive Bayes algorithm applies Bayes’ Theorem with the assumption that features are independent.  
It works well for text because word frequencies are often treated as independent features.

Bayes’ Theorem:
\[ P(A|B) = \frac{P(B|A) * P(A)}{P(B)} \]

In this project:
- \( A \): Email is spam  
- \( B \): Words in the message  

---

## 📊 Results
✅ Model Accuracy: **~90–100%** (on sample data)  
✅ Successfully predicts unseen messages like:  
```
"You have won a free gift card worth $500!" → SPAM
"Are you coming to the meeting tomorrow?" → HAM
```

Confusion Matrix and Classification Report are included in the notebook.

---

## 🧪 Tools & Libraries Used
- **Python 3.x**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**

---

## 🚀 How to Run
1. **Clone this repository**
   ```bash
   git clone https://github.com/asimsheikh-coder/spam-email-detector-using-naive-bayes.git
   cd spam-email-detector-using-naive-bayes
   ```
2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. **Run the notebook**
   ```bash
   jupyter notebook Spam_Email_Detector_Naive_Bayes.ipynb
   ```

---

## 🧑‍💻 Author
**Asim Sheikh**  
12th Grade Student | Aspiring AI Engineer  
📧 Email: asimusmansheikh@gmail.com  
🌐 GitHub: [@asimsheikh-coder](https://github.com/asimsheikh-coder)

---

## 🔖 Citation
> Sheikh, A. *Spam Email Detector using Naive Bayes*. 2025. GitHub Repository.  
> [https://github.com/asimsheikh-coder/spam-email-detector-using-naive-bayes](https://github.com/asimsheikh-coder/spam-email-detector-using-naive-bayes)

---

## 🏁 Conclusion
This project introduces the basics of **text classification** using **Naive Bayes** — a fundamental concept in Natural Language Processing.  
It serves as an ideal starting point for beginners exploring **AI**, **NLP**, and **Machine Learning**.

---
