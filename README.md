# Non-invasive Blood Pressure Estimation via Multi-modal Signal Synchronization

Further details regarding this project are available in the attached Presentation file!
&nbsp;<br>

## 📌 Project Overview
Developed an end-to-end pipeline to estimate Systolic (SBP) and Diastolic (DBP) blood pressure using **Time Series Transformer (TST)** on the vitalDB dataset.



## 🛠️ Key Research Points & Decisions
* **Decision 1: Beat-level Feature Engineering**
  - Synchronized PPG and ECG signals using the **Pan-Tompkins algorithm** to extract **Pulse Arrival Time (PAT)** sequences.
* **Decision 2: Hybrid Architecture (TST + MiniRocket)**
  - Integrated **MiniRocket** as a feature extractor to capture local temporal patterns, which reduced **MAE from 13.67 to 5.51** compared to the baseline TST model.
* **Decision 3: Robustness against Outliers**
  - Adopted **Huber Loss** to mitigate the impact of sensor noise and physiological outliers, achieving a final MAE of **4.87 (SBP) / 4.25 (DBP)**.

## 📊 Performance Visualization
- **Prediction Error Distribution:** Concentrated near zero, confirming high reliability for medical AI applications.
  
<img width="453" height="350" alt="스크린샷 2026-01-06 055828" src="https://github.com/user-attachments/assets/9d6051b7-b709-4793-918e-36a600abbeb0" />
