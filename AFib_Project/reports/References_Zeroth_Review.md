# Reference Papers — Zeroth Review

**Project:** HRV Feature Extraction and Context-Aware False Positive Analysis for Atrial Fibrillation Detection using CACHET-CADB
**Prepared for:** Zeroth Review (Literature Base)
**Format:** IEEE

---

## A. Base Papers (present these 5 in the review)

These four-to-five papers directly define the problem statement, the dataset, the method, and the research gap. Everything else supports them.

**[B1] Dataset — the paper that makes this project possible**
D. Kumar, S. Puthusserypady, H. Dominguez, K. Sharma, and J. E. Bardram, "CACHET-CADB: A Contextualized Ambulatory Electrocardiography Arrhythmia Dataset," *Frontiers in Cardiovascular Medicine*, vol. 9, art. 893090, Jul. 2022. doi: 10.3389/fcvm.2022.893090
*Why it matters:* 259 days of single-lead ambulatory ECG from 24 patients, 1,602 cardiologist-annotated 10-s segments, and — uniquely — synchronized context (activity class, body position, movement acceleration index, step count, stress, sleep). No other public ECG database provides this pairing, which is what makes the false-positive-vs-context analysis possible.

**[B2] Method baseline — HRV + mRMR + classifier**
S. Buś, K. Jędrzejewski, and P. Guzik, "Using Minimum Redundancy Maximum Relevance Algorithm to Select Minimal Sets of Heart Rate Variability Parameters for Atrial Fibrillation Detection," *Journal of Clinical Medicine*, vol. 11, no. 14, art. 4004, Jul. 2022. doi: 10.3390/jcm11144004
*Why it matters:* Shows that only a handful of mRMR-selected HRV parameters separate AF from sinus rhythm across >53,000 60-s ECG segments. This is the direct template for our feature-selection and classification stage.

**[B3] Comparative ML baseline on short ECG segments**
M. S. Jahan, M. Mansourvar, S. Puthusserypady, U. K. Wiil, and A. Peimankar, "Short-term atrial fibrillation detection using electrocardiograms: A comparison of machine learning approaches," *International Journal of Medical Informatics*, vol. 163, art. 104790, 2022. doi: 10.1016/j.ijmedinf.2022.104790
*Why it matters:* Benchmarks multiple classifiers on very short segments (10–20 beats), from the same DTU group behind CACHET-CADB. Gives us the accuracy figures to compare against.

**[B4] SVM + inter-beat-interval baseline**
R. S. Andersen, E. S. Poulsen, and S. Puthusserypady, "A Novel Approach for Automatic Detection of Atrial Fibrillation Based on Inter Beat Intervals and Support Vector Machine," in *Proc. 39th Annu. Int. Conf. IEEE Engineering in Medicine and Biology Society (EMBC)*, 2017, pp. 2039–2042. doi: 10.1109/EMBC.2017.8037253
*Why it matters:* Reported ~96.8% accuracy on MIT-BIH AFDB using RR-interval features and SVM. This is the clinical-lab-condition performance ceiling that our free-living results will be measured against — the gap between the two *is* the research motivation.

**[B5] The problem being solved — false positives in the real world**
M. V. Perez *et al.*, "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," *New England Journal of Medicine*, vol. 381, no. 20, pp. 1909–1917, Nov. 2019. doi: 10.1056/NEJMoa1901183
*Why it matters:* Apple Heart Study — only ~34% of irregular-pulse notifications were confirmed AF. Quantifies the false-positive problem in consumer wearables and justifies the entire context-analysis contribution.

---

## B. Clinical and Dataset Background

[1] G. Hindricks *et al.*, "2020 ESC Guidelines for the diagnosis and management of atrial fibrillation developed in collaboration with EACTS," *European Heart Journal*, vol. 42, no. 5, pp. 373–498, Feb. 2021. doi: 10.1093/eurheartj/ehaa612

[2] S. A. Lubitz *et al.*, "Detection of Atrial Fibrillation in a Large Population Using Wearable Devices: The Fitbit Heart Study," *Circulation*, vol. 146, no. 19, pp. 1415–1424, Nov. 2022. doi: 10.1161/CIRCULATIONAHA.122.060291

[3] A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000. doi: 10.1161/01.CIR.101.23.e215

[4] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, May/Jun. 2001. doi: 10.1109/51.932724

---

## C. ECG Preprocessing and R-Peak Detection

[5] J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm," *IEEE Transactions on Biomedical Engineering*, vol. BME-32, no. 3, pp. 230–236, Mar. 1985. doi: 10.1109/TBME.1985.325532

[6] P. S. Hamilton and W. J. Tompkins, "Quantitative Investigation of QRS Detection Rules Using the MIT/BIH Arrhythmia Database," *IEEE Transactions on Biomedical Engineering*, vol. BME-33, no. 12, pp. 1157–1165, Dec. 1986. doi: 10.1109/TBME.1986.325695

[7] B.-U. Köhler, C. Hennig, and R. Orglmeister, "The principles of software QRS detection," *IEEE Engineering in Medicine and Biology Magazine*, vol. 21, no. 1, pp. 42–57, Jan./Feb. 2002. doi: 10.1109/51.993193

[8] D. Makowski *et al.*, "NeuroKit2: A Python toolbox for neurophysiological signal processing," *Behavior Research Methods*, vol. 53, no. 4, pp. 1689–1696, Aug. 2021. doi: 10.3758/s13428-020-01516-y

---

## D. HRV Feature Theory

[9] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, "Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use," *Circulation*, vol. 93, no. 5, pp. 1043–1065, Mar. 1996. doi: 10.1161/01.CIR.93.5.1043

[10] J. S. Richman and J. R. Moorman, "Physiological time-series analysis using approximate entropy and sample entropy," *American Journal of Physiology — Heart and Circulatory Physiology*, vol. 278, no. 6, pp. H2039–H2049, Jun. 2000. doi: 10.1152/ajpheart.2000.278.6.H2039

[11] S. M. Pincus, "Approximate entropy as a measure of system complexity," *Proc. National Academy of Sciences USA*, vol. 88, no. 6, pp. 2297–2301, Mar. 1991. doi: 10.1073/pnas.88.6.2297

[12] F. Shaffer and J. P. Ginsberg, "An Overview of Heart Rate Variability Metrics and Norms," *Frontiers in Public Health*, vol. 5, art. 258, Sep. 2017. doi: 10.3389/fpubh.2017.00258

[13] M. Brennan, M. Palaniswami, and P. Kamen, "Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability?," *IEEE Transactions on Biomedical Engineering*, vol. 48, no. 11, pp. 1342–1347, Nov. 2001. doi: 10.1109/10.959330

[14] M. Costa, A. L. Goldberger, and C.-K. Peng, "Multiscale entropy analysis of complex physiologic time series," *Physical Review Letters*, vol. 89, no. 6, art. 068102, Aug. 2002. doi: 10.1103/PhysRevLett.89.068102

---

## E. AF Detection from RR Intervals / HRV

[15] D. E. Lake and J. R. Moorman, "Accurate estimation of entropy in very short physiological time series: the problem of atrial fibrillation detection in implanted ventricular devices," *American Journal of Physiology — Heart and Circulatory Physiology*, vol. 300, no. 1, pp. H319–H325, Jan. 2011. doi: 10.1152/ajpheart.00561.2010

[16] S. Dash, K. H. Chon, S. Lu, and E. A. Raeder, "Automatic Real Time Detection of Atrial Fibrillation," *Annals of Biomedical Engineering*, vol. 37, no. 9, pp. 1701–1709, Sep. 2009. doi: 10.1007/s10439-009-9740-z

[17] S. Sarkar, D. Ritscher, and R. Mehra, "A Detector for a Chronic Implantable Atrial Tachyarrhythmia Monitor," *IEEE Transactions on Biomedical Engineering*, vol. 55, no. 3, pp. 1219–1224, Mar. 2008. doi: 10.1109/TBME.2007.903707

[18] A. Petrėnas, V. Marozas, and L. Sörnmo, "Low-complexity detection of atrial fibrillation in continuous long-term monitoring," *Computers in Biology and Medicine*, vol. 65, pp. 184–191, Oct. 2015. doi: 10.1016/j.compbiomed.2015.01.019

[19] S. Hong, Y. Zhou, J. Shang, C. Xiao, and J. Sun, "Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review," *Computers in Biology and Medicine*, vol. 122, art. 103801, Jul. 2020. doi: 10.1016/j.compbiomed.2020.103801

---

## F. Feature Selection, Classification, and Class Imbalance

[20] H. Peng, F. Long, and C. Ding, "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 27, no. 8, pp. 1226–1238, Aug. 2005. doi: 10.1109/TPAMI.2005.159

[21] C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, Sep. 1995. doi: 10.1007/BF00994018

[22] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794. doi: 10.1145/2939672.2939785

[23] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, Jun. 2002. doi: 10.1613/jair.953

[24] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4765–4774.

[25] G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, Oct. 2018. doi: 10.1016/j.neuroimage.2017.06.061
*Supports the subject-wise (patient-independent) cross-validation protocol.*

---

## G. Motion Artifact, Context, and the False-Positive Gap (research-gap evidence)

[26] G. D. Clifford, F. Azuaje, and P. E. McSharry, *Advanced Methods and Tools for ECG Data Analysis*. Norwood, MA, USA: Artech House, 2006, ch. 3 (ECG noise and artifact sources).

[27] J. Yoon, H. Lee, and S. Lee, "Two-stage motion artefact reduction algorithm for electrocardiogram using weighted adaptive noise cancelling and recursive Hampel filter," *PLOS ONE*, vol. 13, no. 11, art. e0207176, Nov. 2018. doi: 10.1371/journal.pone.0207176

[28] Y. Zhang *et al.*, "Artificial Intelligence-Based Atrial Fibrillation Recognition Method for Motion Artifact-Contaminated Electrocardiogram Signals Preprocessed by Adaptive Filtering Algorithm," *Sensors*, vol. 24, no. 12, art. 3789, 2024. doi: 10.3390/s24123789

[29] L. Fan *et al.*, "Heart rate variability and heart rate patterns measured from wearable and implanted devices in screening for atrial fibrillation: potential clinical and population-wide applications," *European Heart Journal*, vol. 44, no. 13, pp. 1105–1117, Apr. 2023. doi: 10.1093/eurheartj/ehac708

[30] A. Bonomi *et al.*, "Atrial Fibrillation Detection Using a Novel Cardiac Ambulatory Monitor Based on Photo-Plethysmography at the Wrist," *Journal of the American Heart Association*, vol. 7, no. 15, art. e009351, 2018. doi: 10.1161/JAHA.118.009351

---

## Notes on the two references currently in the project README that need correction

Reference [4] in `AFib_Project_Full_Explanation.md` attributes the *International Journal of Medical Informatics* paper to "Islam MS, Motin MA." The correct authorship is **Jahan, Mansourvar, Puthusserypady, Wiil, and Peimankar (2022)** — see [B3] above.

Reference [5], "Hasan MI, Motin MA — HRV Feature Extraction and AF Detection Using XGBoost (2025)," could not be verified in any indexed database. Either supply the exact DOI or drop it before the review; an unverifiable citation is the fastest way to lose credibility in a viva.

---

## Suggested framing for the zeroth-review slide

State the gap in one sentence: *AF detectors trained and validated on clinical-grade, motion-free databases (MIT-BIH, CPSC) report 96–99% accuracy [B4], yet in free-living wearable deployment only about one-third of AF alerts are true positives [B5]. No published study has systematically explained this gap by correlating detector errors with the patient's simultaneously recorded physical activity, posture, and movement intensity — a correlation now possible because CACHET-CADB [B1] provides both.*

That sentence is the entire justification for the project, and every reference above exists to defend one clause of it.
