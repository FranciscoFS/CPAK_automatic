# Abstract

## Introduction
Coronal alignment assessment is essential for planning knee surgery. It relies on manually measuring angles such as the hip-knee-ankle angle (HKA), mechanical lateral distal femoral angle (mLDFA), and mechanical medial proximal tibial angle (mMPTA), typically on digital full-length lower-limb radiographs. However, this process is time-consuming, operator-dependent, and prone to measurement error and interobserver variability. This study aimed to develop an artificial intelligence (AI) algorithm to automate these measurements and assess its agreement with manual measurements.

## Methods
A sequential YOLO-based pipeline was developed. First, a detection model localized the hip, knee, and ankle on 280 full-length radiographs, divided into training (n=197), validation (n=55), and test (n=28) sets. A pose-estimation model then identified the anatomical landmarks required to calculate HKA, mLDFA, and mMPTA. The pose model was initially trained on 943 annotated crops and subsequently fine-tuned, using its initial weights, on an expanded dataset of 1,734 crops. The agreement cohort, independent of model training, comprised 57 full-length radiographs. Three observers performed manual measurements. AI-to-observer and interobserver agreement were assessed using the intraclass correlation coefficient (ICC) with 95% confidence intervals (95% CI).

## Results
On the evaluation sets, the detection model achieved an mAP50 of 0.973 and an mAP50-95 of 0.677. The landmark localization model achieved an mAP50 of 0.983 and an mAP50-95 of 0.981. Processing the 57 radiographs required 41.9 seconds (0.74 seconds per image). AI-to-observer agreement was excellent for HKA (ICC=0.984; 95% CI: 0.98–0.99), good for mLDFA (ICC=0.856; 95% CI: 0.80–0.90), and good for mMPTA (ICC=0.819; 95% CI: 0.74–0.87). Bias was below 0.6° for all three measurements.

## Discussion and conclusion
Sequential training and progressive fine-tuning yielded high performance for anatomical detection and landmark localization, with a processing time compatible with clinical use. Agreement between the AI and manual measurements was excellent for HKA and very good for mLDFA and mMPTA, with minimal bias across all measurements. These findings suggest the tool could provide objective and efficient support for preoperative coronal alignment planning.
