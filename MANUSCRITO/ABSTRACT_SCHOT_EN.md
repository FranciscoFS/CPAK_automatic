**Development and Validation of a Deep Learning Algorithm for Automated Coronal Alignment Measurement in Long Limb X-rays.**

**Authors:** Francisco Fernandez, Raúl Zilleruelo, Joaquín Steinsapir, Catalina Vidal, Cristian Ruz, Pablo Besa.

**Introduction:** Coronal alignment assessment is a fundamental step in planning knee surgery, guiding decisions regarding corrective osteotomies and arthroplasty. It relies on measuring angles on long-limb radiographs (LLR), including the hip-knee-ankle angle (HKA), the mechanical lateral distal femoral angle (mLDFA), and the mechanical medial proximal tibial angle (mMPTA). These measurements characterize the coronal profile (varus, neutral, or valgus) and help phenotype patients for surgical planning. However, this process is time-consuming, requires training, and is subject to interobserver variability.

**Objective:** To develop and validate a deep learning algorithm for automated coronal alignment measurement in LLR, compared against expert measurements.

**Methods:** A deep learning algorithm based on open-source computer vision models (YOLO) was developed and organized into three stages (Fig. 1): localization of the hip, knee, and ankle; identification of anatomical landmarks; and calculation of HKA, mLDFA, and mMPTA. The algorithm was trained on 280 radiographs for joint detection and 1,734 anatomical crops for landmark localization, with 80% used for training and 20% for validation. For external validation, its measurements were compared with those of two experts on 112 limbs from 57 LLR, independently of training. Agreement was assessed using the intraclass correlation coefficient (ICC), with 95% confidence intervals (95% CI), and the sample size was calculated for two raters, α=0.05, 80% power, and an expected ICC of 0.90 against a minimum acceptable of 0.70; 19 subjects were required. Our 112 limbs provide >99% power to detect ICCs above 0.70 across all measures. Bias was evaluated through Bland-Altman analysis.

**Results:** The model achieved a mean average precision (mAP50) >97% for joint detection and >98% for landmark localization, with a processing time of 0.72 s per radiograph. Agreement between the model and expert measurements was excellent for HKA (ICC=0.995; 95% CI: 0.99–1.00) and good for mLDFA (ICC=0.873; 95% CI: 0.83–0.91) and mMPTA (ICC=0.873; 95% CI: 0.80–0.92). Interobserver agreement between the two experts was excellent for HKA (ICC=0.997; 95% CI: 0.99–1.00), mLDFA (ICC=0.930; 95% CI: 0.87–0.96), and mMPTA (ICC=0.928; 95% CI: 0.89–0.95). Bland-Altman analysis showed biases of 0.20°, −0.15°, and −0.81° for HKA, mLDFA, and mMPTA, respectively (Fig. 2). The mMPTA had the largest bias and dispersion, consistent with its recognized difficulty of measurement, even among expert surgeons.

**Conclusion:** The model showed excellent performance and processing speed, with high agreement and low bias compared to expert measurements, approaching the level of interobserver agreement between the two experts. These findings suggest that AI-based models can standardize and optimize coronal alignment analysis in LLR.

**Figures**

**Figure 1:** Algorithm flowchart; from the input radiograph to angle calculation. HKA = Hip-Knee-Ankle angle; mLDFA = mechanical lateral distal femoral angle; mMPTA = mechanical medial proximal tibial angle.

**Figure 2 Left column:** Scatter plots of HKA, mLDFA, and mMPTA measurements from the model versus the observers' average. **Right column:** Bland-Altman plots for the same measurements.
