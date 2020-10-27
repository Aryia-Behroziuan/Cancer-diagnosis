# Cancer-diagnosis
Diagnosis of cancer with artificial intelligence

Abstract
Cancer is an aggressive disease with a low median survival rate. Ironically, the treatment process is long and very costly due to its high recurrence and mortality rates. Accurate early diagnosis and prognosis prediction of cancer are essential to enhance the patient's survival rate. Developments in statistics and computer engineering over the years have encouraged many scientists to apply computational methods such as multivariate statistical analysis to analyze the prognosis of the disease, and the accuracy of such analyses is significantly higher than that of empirical predictions. Furthermore, as artificial intelligence (AI), especially machine learning and deep learning, has found popular applications in clinical cancer research in recent years, cancer prediction performance has reached new heights. This article reviews the literature on the application of AI to cancer diagnosis and prognosis, and summarizes its advantages. We explore how AI assists cancer diagnosis and prognosis, specifically with regard to its unprecedented accuracy, which is even higher than that of general statistical applications in oncology. We also demonstrate ways in which these methods are advancing the field. Finally, opportunities and challenges in the clinical implementation of AI are discussed. Hence, this article provides a new perspective on how AI technology can help improve cancer diagnosis and prognosis, and continue improving human health in the future.

Previous article in issueNext article in issue
Keywords
Cancer diagnosisPrognosis predictionDeep learningMachine learningDeep neural network
1. Introduction
In 2019, the cancer burden is estimated 1.7 million new cases and 0.6 million deaths in the United States [1].As the incidence rate of cancer and its mortality have risen sharply, prolonging survival and reducing local recurrence have become increasingly dependent on modern laparoscopy surgery, robotic surgery, tumor adjuvant therapy, and other new technologies [2]. Treatment for cancer currently involves various options [[3], [4], [5]]. Since the 2010s, the effectiveness of cancer treatment has improved significantly [[1], [2], [3], [4], [5]]. However, despite the profusion of new techniques, scientifically satisfactory curative results for each stricken individual are elusive due to uncertainties in diagnostic precision. Thus, patient-specific optimal treatment could be adopted if an accurate prognosis could be made. Indeed, improvements in prediction accuracy could greatly assist doctors in planning patient treatments and eliminating both the physical and mental hardships brought on by the disease. Fundamental clinical observations can be combined with the implementation of the traditional TNM staging approach (based on the size of the tumor (T), the spread of the cancer into nearby lymph nodes (N), and the spread of the cancer to other body parts (M, for metastasis)) in empirical tests, but erroneous predications of prognoses continue to pose a bottleneck for clinicians [10]. Improvement in prognosis accuracy using state-of-art AI technology remains a critical challenge for clinical researchers.

Technical advancements in statistics and computer software have enabled computer engineers and health scientists to collaborate closely toward prognosis improvements using multi-factor analysis, conventional logistic regression, and Cox analyses. The accuracy of such predictions was found to be significantly higher than that of empirical predictions. With the implementation of AI, scholars have lately turned to establishing models using AI algorithms to predict and diagnose cancer. These methods currently play a major role in improving the accuracy of cancer susceptibility, recurrence, and survival predictions.

Cancer is difficult to diagnose at early stages or can easily relapse after treatment. Moreover, accurate predictions of disease prognosis with high certainty are very difficult. Some cancers are difficult to detect in their early stages due to their vague symptoms and the indistinctive tell-tale signs on mammograms and scans. Thus, developing better predictive models using multi-variate data and high-resolution diagnostic tools in clinical cancer research is imperative. A quick search of the literature shows that the number of papers on cancer analysis has grown exponentially, especially those involving AI tools and large data sets containing historical clinical cases for training AI models [6]. Moreover, the literature reports that traditional analysis methods such as statistical analysis and multivariate analysis are not as accurate as AI. This is particularly true when AI is used in conjunction with sophisticated bioinformatics tools, which can significantly improve the diagnostic, prognostic, and predictive accuracies [[7], [8], [9], [10]]. A more specific concept, namely machine learning (ML), is becoming increasingly prevalent. ML is a subset of AI, and is used to construct predictive models that learn logical patterns from mass historical data so as to predict the survival rate of a patient. ML has been used extensively for improving prognosis [[11], [12], [13]]. Prognostication is an important clinical skill, particularly for clinicians working with cancer patients [14,15]. ML methods have been shown to improve the accuracy of cancer susceptibility, recurrence, and survival predictions, three aspects that are fundamental to early diagnosis and prognosis in cancer research. ML can provide good results with regard to the clinical management of patients [[16], [17], [18], [19], [20], [21], [22]]. This aspect has motivated researchers in the biomedical and bioinformatics fields to develop more effective ML tools that can classify cancer patients into high- or low-risk recurrence groups for refined prognosis management.

Over the years, AI has been widely used in clinical cancer research due to its feasibility and advantages. This study selected and analyzed relevant publications from the PubMed, Google Scholar, CNKI, and WANFANG databases between 1995 and 2019. In total, using matching keywords, 3594 papers in these databases were found to be related to AI studies on cancer. Of these, 1136 papers were found to be duplicate and excluded. leaving a total of 2458 papers. These papers were further examined for relevance using their titles/abstracts, and 2365 papers were found to be relevant. Using a forward citation search, we included 126 full-text papers on the use of AI for cancer diagnosis and prognosis predictions. This process is illustrated in Fig. 1. The majority of the papers (66) were published in the United States, followed by China (40). Most of those papers contain keywords such as “machine learning,” “deep learning,” “artificial intelligence,” and “prognosis.” Our comprehensive survey provides a systematic literature review on the use of AI in cancer research, and provides important insights into the applications of AI to cancer diagnosis and prognosis in order to spearhead cutting edge research in this field.

Fig. 1
Download : Download high-res image (472KB)Download : Download full-size image
Fig. 1. Process of searching and selecting primary studies for the review conducted in this study.

2. Basics of artificial intelligence
The concept of AI first emerged in 1956, the aim being to build machines that can think and reason over complex tasks just like human beings and thereby sharing the same essential cognitive characteristics. Since then, the field of AI has made many developments as AI theories and implementation gradually became reality in scientific research laboratories. Fig. 2 illustrates the milestones in the field of AI research, as well as compares the advantages and disadvantages of the different working principles of ML and deep learning (DL). These methods have different basic characteristics and can be applied in various fields. In recent years, researchers developed these approaches and introduced the broad learning system for application in cancer precision medicine. We summarize these AI methods in this review and systematically show the differences among them and their applications in cancer analysis by clinical specialists. AI research is now expanding into various sub-branches, such as expert systems, ML, evolutionary computing, fuzzy logic, computer vision, natural language processing, and recommendation systems. Fig. 2 shows the branches of AI application in clinical research. Fundamentally, ML uses algorithms to parse data, learn underlying patterns, and offer insights using which decisions and predictions about real-world events can be made. Unlike traditional hard-coded software programs that solve specific tasks, ML uses large amounts of data to “train” and apply algorithms to dynamically learn how certain tasks can be accomplished. DL is not an independent learning method. Two modes, supervised and unsupervised learning methods, are available to train a deep neural network. The exponential development of this field in recent years has given rise to unique learning methods such as the residual network. Thus, increasingly, DL is now being regarded as a learning method alone. However, simply put, ML is used to realize AI while DL is used to implement ML. Despite all of its advances, the following limitations remain with regard to DL: 1) DL models require a large amount of training data to induce an accurate model. However, in real life, certain biomedical samples may only exist in small quantities. 2) In some fields, traditional and simple ML methods can be used to solve problems; complex DL methods are not required.

Fig. 2
Download : Download high-res image (1MB)Download : Download full-size image
Fig. 2. The concepts of AI, ML, and DL, and their relationships. This schematic outlines the milestone in AI research, and compares the advantages and disadvantages as well as the different working principles of ML and DL. a) ML and DL are used to implement AI, and DL is a technology used to implement ML. AI, ML, and DL were first discovered in the 1950s, 1980s, and 2010s, respectively. b) Features of and differences between ML and DL with regard to features engineering, execution time, interpretability, and data and hardware dependency. c) ML relies on engineered features extracted from specific regions on the basis of expert knowledge. The workflow comprises the following steps: data preprocessing after acquisition, features extraction, and selection and classification of algorithms. d) DL uses localization for features engineering without region annotations. It comprises several layers in which features extraction, selection, and ultimate classification are performed simultaneously during training. Both methods are data-centric and involve AI, which has been applied to the field of cancer research.

3. Artificial intelligence in clinical cancer prognosis prediction
In the last few decades, various types of clinicians, ranging from specialty doctors to paramedics, have been called upon to make clinical cancer prognosis predictions based on their work experience. With the arrival of the digital information era, clinicians understand the importance of using AI technology such as ML and DL as a decision support tool (Fig. 2a). Many differences exist between ML and DL (Fig. 2b). For instance, ML (Fig. 2c) involves more processes than DL (Fig. 2d). In our previous study, we used AI to predict brain metastasis [23]. AI is thus being increasingly applied toward clinical cancer prognosis. Given the failure of traditional statistical analysis to provide accurate predictions, it is difficult to predict how a patient's cancer will progress. Clinicians are also concerned about patients' risk of contracting the disease, tumor recurrence, or death after treatment. Such aspects are highly related to the choice of treatment and curative effects. In fact, most research on clinical cancer is currently aimed at determining the prognosis or predicting the correct outcome in response to treatment. If the prognoses of different patients can be predicted more accurately, more precise and suitable treatments can be provided to them; in fact, such treatments tend to be individualized or customized to patients. To date, accurate treatment customized for a patient is very difficult to implement. However, AI can be used to process and analyze multi-factor data from multiple patient examination data to predict cancer prognosis as well as the survival time and the disease progress of patients more accurately.

While medical statistics are commonly used for cancer prognostics, the use of AI for the same task tends to be less common. In this paper, we present a review of the literature on application of AI to various kinds of cancer prognoses by global scientists in different study populations (Table 1). In the last decade, the number of such studies have increased rapidly in China, the United States, and Europe. Usually, medical statistics cover methods such as area under the curve (AUC). Cancer prognosis involves predictions of disease recurrence and patient survival, the aim being to improve patient management [24,25]. Enshaei A et al. [26] compared a variety of algorithms and classifiers with conventional logistic regression statistical approaches to demonstrate that AI may have a role in providing prognostic and predictive data for ovarian cancer patient. Khan et al. [27] used a number of decision tree rules, fuzzy membership functions, and inference techniques for breast cancer survival analysis. Their performance comparisons suggested that predictions of weighted fuzzy decision trees (wFDT) are more accurate and balanced than independently applied crisp decision tree classifiers. Moreover, this approach showed good potential for significant cancer prognosis prediction performance enhancement (seeTable 2).

Table 1. AI applied to various kinds of cancer prognosis by the global scientist in different study population.

Type of Cancer	Authors	Year	Country/Region	Number of Patients in Study	Age (Years)	Study Population	Methods	Results
Prostate Cancer	Kuo et al. [37]	2015	Taiwan	100	75	Hospital	Fuzzy Neural Network	/
Zhang et al. [38]	2017	USA	/	/	TCGA	SVM model	Average Accuracy (66%)
Breast Cancer	Sun et al. [31]	2018	China	1980	61		Multimodal DNN	/
Park et al. [33]	2013	USA	162500	N/A	SEER	Semi-supervised Learning Model	/
Delen et al. [34]	2005	USA	433272	60.61	SEER	ANN and DT	Accuracy: DT (93.6%), ANN (91.2%)
Lu et al. [39]	2019	USA	82707	58.38	SEER	Dynamic Gradient Boosting Machine with GA	Accuracy Improved (28%)
Glioblastoma	Vasudevan et al. [40]	2018	India	215	N/A	TCGA	Neural Network	Accuracy: DT (89.2%)
Bladder Cancer	Tian et al. [41]	2019	China	115	N/A	Hospital	Statistical Analysis	NEDD8: Poor Prognosis Found
Hasnain et al. [42]	2019	USA	3503	67.8	Hospital	KNN, RF, etc	Sensitivity& Specificity (>70%)
Nasopharyngeal Carcinoma	Zhang et al. [21]	2019	China	3269	45	Hospital	Large Scale, Big Data Intelligence Platform	EBV DNA: a Robust Biomarker for NPC Prognosis
Gastric Cancer	Biglarian et al. [43]	2011	Iran	436	58.43 ± 13.02	Hospital	Cox Proportional Hazard, ANN	TP(83.1%),
Zhu et al. [44]	2013	China	289	63.20 ± 10.75	Hospital	ANN	TP: ANN(85.3%)
Colorectal Cancer	Bottaci et al. [45]	1997	UK	334	N/A	Hospital	Six Neural Networks	Accuracy(>80%), mean Sensitivity(60%),
mean Specificity(88%)
Wang et al. [46]	2019	China	1568	N/A	SEER	Semi-random Regression Tree	/
Bychkov et al. [47]	2018	Finland	641	N/A	Hospital	LSTM, Naïve Bayes, SVM	Hazard Ratio(2.3); CI(95%,1.79–3.03), AUC(0.69)
Oral Cancer	Chang et al. [48]	2013	Malaysia	156	N/A	MOCDTBS	Hybrid model of ReliefF-GA-ANFIS	Accuracy(93.81%),AUC (0.9)
Lung Cancer	Lynch et al. [49]	2017	USA	10442	N/A	SEER	GBM, SVM	RMSE(32,15.05) for GBM, SVM
Sepehri et al. [50]	2018	France	101	N/A	Hospital	SVM with RFE and RF	Accuracy(71%, 59%)
Yu et al. [51]	2016	Italy	168	N/A	Hospital	Naive Bayes, SVM with Gaussian, etc	/
Ovarian Cancer	Lu et al. [52]	2019	Taiwan	84	59.94 ± 11.25	Both	SVM	HR(0.644), CI(95%,0.436–0.952)
Lu et al. [53]	2019	UK	364	N/A	Both	Unsupervised Hierarchical Clustering	RPV: A Novel Prognostic Signature Discovered
Acharya et al. [54]	2018	Singapore&
Malaysia	469	23–90	Hospital	Fuzzy Forest	Accuracy(80.60 ± 0.5%), Sensitivity(81.40%), Specificity (76.30%)
Glioma	Lu et al. [55]	2018	Taiwan	456	N/A	TCGA	Improved SVM	Accuracy(81.8%), ROC(0.922)
Papp et al. [56]	2018	Austria	70	48 ± 15	Hospital	GA and Nelder–Mead ML methods	Sensitivity (86%–98%), Specificity (92%–95%)
Spinal Chordoma	Karhade et al. [57]	2018	USA	265	N/A	SEER	Boosted DT, SVM, ANN	5-year Survival (67.5%)
Long Bone Metastases	Stein et al. [58]	2015	USA	927	62 ± 13	Hospital	Multiple Additive Regression Trees	/
Oral Cavity Squamous Cell	Lu et al. [59]	2017	USA	115	61.0 ± 12.	Hospital	RF, SVM	AUC(0.72), Accuracy(70.77), Specificity(73.08), Sensitivity(61.54)
Pancreatic Neuroendocrine	Song et al. [122]	2018	China	8422	59(48–69)	SEER	SVM, RF,DL	Accuracy(81.6% ± 1.9%),curve(0.87)
*SVM: Support Vector Machine, DNN: Deep Neural Network, ANN: Artificial Neural Network, DT: Decision Tree, GA: Genetic Algorithm Optimizer, KNN: K-Nearest Neighbor, RF: Random Forest, LSTM: Long Short-Term Memory Network, GBM: Gradient Boosting Machines, RFE: Recursive Feature Elimination, TP: True Prediction.


Table 2. AI applied to various kinds of cancer prognosis by the global scientist in different study population.

Authors	Year	Country/Region	Number of patients in the study	Study population	Methods	Results
Li et al. [70]	2019	China	17627	Both	DCNN	Sensitivity(93.4%), CI(95%,89.6–96.1) Specificity(86.1%,p < 0.0001)
Esteva et al. [89]	2017	USA	2032	Both	Inception v3 CNN	AUC (over 91%)
Zhu et al. [76]	2019	China	203	Hospital	CNN	Sensitivity(76.47%), and Specificity(95.56%), Overall Accuracy(89.16%),CI(95%,90–97)
Samala et al. [77]	2018	USA	2566	Both	DCNN	AUC(0.85 ± 0.05)
Coudray et al. [81]	2018	USA	137	TCGA,NCI Genomic Data Commons	DCNN(inception v3)	AUC(0.733–0.856)
Wu et al. [109]	2018	Italy	1034	Hospital	Bayesian network	/
Yi et al. [85]	2018	Italy	436	Hospital	Decision Tree J48	Accuracy (80%)
Stephan et al. [87]	2002	Germany	928	Hospital	ANN	Specificity Level (90%)
Lorenzo et al. [90]	1999	Italy	98	Hospital	Multivariate Cluster Analysis	/
Nadia et al. [91]	2019	Italy	374	Lymphoma and IDC Datasets	Convolutional Autoencoder, Supervised Encoder FusionNet	F-measure Score Improved (5.06%), Accuracy Improved (5.06%)
Tabibu et al. [110]	2019	India	Ensemble	TCGA	CNN	Accuracy (92.61%)
Haenssle et al. [60]	2018	International Skin Imaging Collaboration (ISIC)	100	International Skin Imaging Collaboration (ISIC)	Google's Inception v4 CNN architecture	Sensitivities(86.6%–88.9%, ROC AUC(>0.86,P < 0.01)
DCNN:Deep Convolutional Neural Network, ANN: Artificial Neural Network, AUC: Area Under the Curve, Ensemble:1027 (KIRC), 303 (KIRP), and 254 (KICH) tumor slide images.


3.1. Breast cancer prognosis prediction
Breast cancer prognosis involves estimation of the recurrence of disease and predicting the survival of the patient, hence resulting in improved patient management. Researchers often use multimodal deep neural networks (DNNs) by integrating multi-dimensional data to compare of the receiver operating characteristic (ROC) curve and AUC values. The results indicate that combining different data types and ensemble DNN methods is an efficient way to improve human breast cancer prognosis predictions (Fig. 3). Jhajharia et al. [28] applied principal component analysis by preprocessing the data and extracting features in the most relevant form for training artificial neural networks (ANNs) to learn patterns in the data for classification of new instances. Data- and learning-based approaches can provide an effective framework for prognostic research by accurately classifying data instances into relevant categories based on tumor severity. Ching et al. [29] developed a new ANN framework called Cox-nnet (a neural network extension of the Cox regression model) to predict patient prognoses from high throughput transcriptomics data. Cox-nnet reveals much richer biological information, at both the pathway and gene levels, by analyzing features represented in the hidden layer nodes in Cox-nnet. Bomane A et al. [30] applied three classifiers features selected to individually link to the cytotoxic-drug sensitivities and prognosis of patients on breast cancer for optimizing paclitaxel-therapies in clinical practice.Sun et al. [31] proposed a multimodal DNN by integrating multi-dimensional data (MDNNMD) for the prognosis prediction of breast cancer. The novelty of their method lies in the design of the method's architecture and the fusion of multi-dimensional data. The results of the comprehensive performance evaluation showed that the proposed method outperformed all the other prediction methods using single-dimensional data.

Fig. 3
Download : Download high-res image (504KB)Download : Download full-size image
Fig. 3. Human breast cancer prognosis prediction using multimodal DNNs by integrating multi-dimensional data including gene expression profile, copy number alteration (CNA) profile, and clinical data. The prediction model consists of a triple modal DNN and finally combined predictive scores from each independent model. a and b [31] show a comparison of the ROC curve and AUC value. The results indicated that combining different data types and ensemble DNN methods is an efficient way to improve human breast cancer prognosis prediction performance.

Chi et al. [32] applied ANNs to survival analysis as ANNs can easily consider variable interactions and create a non-linear prediction model thus offering more flexible predictions of survival time than traditional methods. Their study compared the results of the ANNs for two different breast cancer datasets, both of which contained nuclear morphometric features. The results showed that ANNs can successfully predict recurrence probabilities and separate patients with good (more than five years) and bad (less than five years) prognoses. Park et al. [33] suggested that a semi-supervised learning model can be easily applied by medical professionals without expending the time and effort for parameter searching in conventional models. The ease of use and lowered search time will eventually lead to more accurate and less-invasive prognoses for breast cancer patients. Delen et al. [34] used ANNs and decision trees along with a traditional statistical method (logistic regression) to develop prediction models using more than 200,000 cases. They found that the decision tree (C5) was the best predictor, showing 93.6% accuracy on the holdout sample, followed by ANNs with 91.2% accuracy, and logistic regression models with the worst accuracy (89.2%). Sun et al. [35] and Gevaert et al. [36] proposed a more practical strategy that utilized both clinical and genetic marker information, which may be complementary given the complexity of breast cancer prognosis.

The hybrid signature performed significantly better than other methods, including the 70-gene signature, clinical makers alone, and the St. Gallen consensus criterion. At the 90% sensitivity level, the hybrid signature achieved 67% specificity as compared to 47% for the 70-gene signature and 48% for the clinical makers. The odds ratio of the hybrid signature for developing distant metastases within five years between patients with a good prognosis signature and those with a bad prognosis was 21.0 (95% CI: 6.5–68.3), far higher than that of either the genetic or the clinical markers alone. Xing et al. [61] presented a general clustering-based approach called algorithm of clustering of cancer data(ACCD) to develop a predictive system for cancer patients. Xu et al. [62] adopted an efficient feature selection method, known as the support vector machine-based recursive feature elimination (SVM-RFE), for gene selection and prognosis prediction. Using the leave-one-out evaluation procedure on a gene expression dataset including 295 breast cancer patients, they discovered a 50-gene signature, which could be combined with SVM to achieve a superior prediction performance showing an improvement of 34, 48, and 3% in accuracy, sensitivity, and specificity, respectively, compared with the widely used 70-gene signature. Further analysis showed that the 50-gene signature was effective at predicting the prognoses of metastases and distinguishing patients who should receive adjuvant therapy. Lu et al. [39] showed that their proposed a genetic algorithm-based online gradient boosting (GAOGB) model achieved statistically outstanding online learning effectiveness. Rohit et al. [63] used three different machine learning methods to predict breast cancer survivability separately for each stage, and compared them with the traditional joint models for all the stages. Wang et al. [64] showed that the synthetic minority over-sampling technique + particle swarm optimization + C5 (SMOTE + PSO + C5) hybrid algorithm is the best among all algorithm combinations for 5-year survivability of breast cancer patient classification. Shukla et al. [65] developed a robust data analytical model to help improve understanding of breast cancer survivability in presence of missing data. Beibit et al. [66] proposed a neural network-based entity embedding approach to acquire continuous vector representations of categorical variables to interpret categorical variables and improve prognosis using classifiers.

Medical images trained with deep learning can further improve the accuracy of cancer re-staging. Using big data from images to establish the prognostic model, we can acquire a superior prediction of cancer patient prognosis. Sepehri et al. [50] compared two ML pipelines to build prognostic models exploiting clinical and 18F-FDG PET/CT radiomics features in lung cancer patients. 18F-FDG PET/CT is a combination of positron emission tomography (PET) with 18F-labeled fluoro-2-deoxyglucose (18F-FDG) and computed tomography (CT). They showed that although SVM provided better accuracy than RF in the training step, RF had the highest validation performance (71% vs. 59%). Cao et al. [67] evaluated 168 patients affected by ovarian carcinoma, who underwent a restaging 18F-FDG PET/CT. The increased odds ratio was assessed using Cox regression analysis testing of all lesion parameters measured by PET/CT, and the results indicated that 18F-FDG PET/CT has important prognostic value in assessing the risk of disease progression and mortality rate.

3.2. Gastric cancer prognosis prediction
ANN has been shown to be a more powerful statistical tool for predicting the survival rate of gastric cancer patients compared to the Cox proportional hazard regression model. Oh et al. [68] used a survival recurrent network (SRN) to predict survival, and the results corresponded closely with actual survival. Thus, the SRN model was more accurate at survival prediction than the staging defined by the American Joint Committee on Cancer (AJCC). While TNM staging is a grouped prediction considering only tumor factors, the SRN model provides an individualized prediction based on numerous prognostic factors; basically, patient grouping is not necessary. Biglarian et al. [56], analyzed 436 registered gastric cancer patients who had had surgery between 2002 and 2007 at the Taleghani Hospital, Tehran (Iran), to predict the survival time using Cox proportional hazard and ANN techniques. The estimated one-, two-, three-, four-, and five-year survival rates of the patients were 77.9, 53.1, 40.8, 32.0, and 17.4%, respectively. The Cox regression analysis revealed that the age at diagnosis, high-risk behaviors, extent of wall penetration, distant metastasis, and tumor stage were significantly associated with the survival rates of the patients. The true prediction of the neural network was 83.1%, and the corresponding value for the Cox regression model was 75.0%.

Another study [57] demonstrated that the ANN model is a more powerful tool in determining significant prognostic variables for gastric cancer patients, which are recommended for determining the risk factors of such patients. Maroufizadeh et al. [73,74] showed that the neural network model is a more powerful tool in determining important variables for gastric cancer patients compared to the conventional statistical method (Weibull regression model) [75,76]. Alexander et al. [77] predicted disease-specific gastric cancer survival at a European institution by using a U.S.-derived nomogram. Amiri et al. [78] assessed the application of neural networks to survival analysis in comparison to the Kaplan–Meier and Cox proportional hazards models.

4. Artificial intelligence in cancer diagnosis
Clinicians usually rely on their personal knowledge and clinical experience when examining patients’ signs and symptoms. This clinical information and data can be used to diagnose disease, but the accuracy of the diagnosis cannot be guaranteed, and it is impossible to avoid mistaken diagnoses. This aspect points to the limited ability of the human brain to integrate large amounts of sample data. However, AI models are extremely adept at handling vast amounts of data. Integrative processing and extraction can allow more accurate disease diagnosis due to the efficiency and effectiveness of learning and training large samples (Fig. 4). Their practicality and accuracy are also higher than those of expert diagnoses. DL refers to a set of computer models that have recently been used to make unprecedented progress in the way computers extract information from images. DL algorithms have been applied to tasks in numerous medical specialties (most extensively, radiology and pathology), and in some cases, they have attained performance comparable to that of human experts. Furthermore, it is possible that DL could be used to extract information from medical images that would not be immediately apparent by human analysis alone, and that could be used to inform on molecular status, prognosis, or treatment sensitivity [69]. A performance comparison of prognosis and diagnosis between AI methods and pathologists is shown in Fig. 4. The diagnostic performance of CNNs was found to be superior to that of most but not all dermatologists.

Fig. 4
Download : Download high-res image (228KB)Download : Download full-size image
Fig. 4. Performance comparison of prognosis and diagnosis between AI methods and human pathologists. a) Bychkov et al. [47] trained a deep learning-based classifier to predict five-year disease-specific survival in a comprehensive series of digitized tumor tissue samples of CRC stained for basic morphology. Images of H&E stained TMA spots were obtained from 420 patients who survived and died of colorectal cancer within five years after diagnosis. The authors compared the prognosis made by the computer and the pathologists. The long short-term memory (LSTM) network, a DL method was assessed, and TMA spot and whole-slide level was performed by human experts. The machine-based prognosis could extract more prognostic information from the tissue morphology of colorectal cancer than an experienced human observer. b) Haenssle et al. [60] aimed to facilitate melanoma detection by comparing the diagnostic performance of a CNN with that of a large group of 58 international dermatologists, including 17 “beginners,” 11 “skilled,” and 30 “expert” doctors. The levels were classified as follows: Level-I involved dermoscopy only, while Level-II involved dermoscopy along with clinical information and images. The diagnostic performance of the CNN was superior to most, but not all, dermatologists. The ROC of the AUC (0.86) for the CNN was greater than the mean ROC area of the dermatologists (0.82 and 0.79, respectively; P < 0.01). Thus, physicians having different levels of training and experience may benefit by tapping into the image classification abilities of the CNN.

4.1. Solid tumor diagnosis
Recently, the use of a deep convolutional neural network (DCNN) model has been shown to improve the diagnostic accuracy of thyroid cancer by analyzing sonographic imaging data from clinical ultrasounds [70]. The DCNN model showed similar sensitivity and improved specificity with regard to identifying patients with thyroid cancer compared to a group of skilled radiologists. The improved technical performance of the DCNN model warrants further investigation via randomized clinical trials. Hu et al. [71] believed that DL models can broadly influence clinical practice. Another study [72] developed and validated DCNN algorithms using the largest number of images to date. Yet, the accuracy in three small-scale validation sets was not satisfactory as it ranged from 0·857 to 0·889. Another study [73] showed that the technical performance of AI models should be thoroughly validated in different geographic settings. Mori et al. [74] expect giant leaps in AI applications to gastrointestinal endoscopy in the following ten years. A convolutional neural network computer-aided detection (CNN-CAD) system based on endoscopic images was constructed to determine invasion depth and screen patients for endoscopic resection. The results showed high accuracy and specificity, allowing early gastric cancer to be distinguished from deeper submucosal invasion, and minimized overestimation of invasion depth to reduce unnecessary gastrectomies. Ichimasa K et al. [75] believed artificial intelligence significantly reduced unnecessary additional surgery after endoscopic resection of T1 colorectal cancer (CRC) without missing lymph node metastasis (LNM) positivity. A previous study [76] used a DCNN model for classification of malignant and benign masses in digital breast tomosynthesis (DBT). A multi-stage transfer learning approach utilizing data from similar auxiliary domains was also tested for intermediate-stage fine-tuning [77]. Combined deep belief networks (DBNs) with extreme learning machine (ELM) classifiers can be used to fine-tune the network weights and biases, and when combined with a genetic algorithm (GA), they can find a suitable number of hidden layers and neurons to promote diagnostic performance in the classification of breast cancer. Automatic classification of perifissural nodules can make lung cancer screening more efficient and reduce the number of follow-up visits. The results showed that the performance of this approach (AUC: 0.868) was close to that of human experts [78]. An AdaBoosted back propagation neural network (BPNN) using each feature type and fusing the decisions made by three classifiers to differentiate nodules could achieve AUC values of 96.65%, 94.45%, and 81.24%, respectively, which was substantially higher than that obtained by other approaches [79]. A novel automated pulmonary nodule detection framework with a 2D convolutional neural network (CNN) was used to assist the CT reading process [80]. The nodule candidate detection sensitivity was 86.42%. For the false positive reduction, the sensitivity reached 73.4% and 74.4% at 1/8 and 1/4 FPs/scan, respectively. This result illustrates that the proposed method could facilitate accurate pulmonary nodule detection.

Nicolas et al. [81] trained a DCNN on whole-slide images obtained from The Cancer Genome Atlas to accurately and automatically classify them as lung adenocarcinoma (LUAD), lung squamous cell carcinoma (LUSC), or normal lung tissue. The results showed that 6 of 10 most commonly mutated genes in LUAD (STK11, EGFR, FAT1, SETBP1, KRAS and TP53) could be predicted from pathology images, with AUC values ranging from 0.733 to 0.856, as measured on a held-out population. Nam et al. [82] proposed a DL-based automatic detection algorithm, which outperformed physicians in radiograph classification and nodule detection of malignant pulmonary nodules on chest radiographs. The technique enhanced physicians’ performances when used as a second reader. Other studies [83,84] used Bayesian network meta-analysis to simultaneously re-evaluate efficacy and safety. The results indicated certain prior distributions that yielded posterior distributions of the parameters of interest, thus allowing clinicians and health policy makers to make more informative decisions. Yi et al. [85] described that ML-based quantitative texture analysis (QTA) can differentiate subclinical pheochromocytoma (sPHEO) from lipid-poor adenoma (LPA) when adrenal incidentaloma is present. Romeo et al. [86] used the J48 classifier as the feature selection method and obtained a diagnostic accuracy of 80% in adrenal lesions on unenhanced MRIs, which was an improvement over that of the expert radiologist (73%). Another study [87] used an ANN to improve the risk assessment of prostate cancer. The findings showed a sensitivity level of 90% and an enhanced specificity by 15–28%. Thus, the results showed that despite large amounts of input data, ANNs show promise in decreasing the number of false positives when detecting prostate cancer. With regard to classifying skin cancer, CNNs [88] can achieve performance on par with that of all tested experts, demonstrating the value of AI for such tasks [89].

4.2. Non-solid tumor diagnosis
The results of cluster and discriminant analyses for various types of Non-Hodgkin lymphomas (NHLs) reveal that a combination of proliferation-associated parameters rather than a single one facilitates better distinctions between groups of lymphomas with unequal growth characteristics in non-solid tumors [90]. The use of DL in the automatic analysis of hematoxylin- and eosin-stained histological images resulted in an F-measure score of 5.06% in the detection task and an improvement of 1.09% in the accuracy measure for the classification task [91]. Moreover, a DL algorithm called LYmph Node Assistant or LYNA could detect metastatic breast cancer in sentinel lymph node biopsies, thus improving the pathologist's productivity and reducing false negatives [92]. Haenssle et al. [60] compared the diagnostic performance of a CNN with that of 58 dermatologists (30 of whom were experts). Most dermatologists performed more poorly than the CNN (Fig. 4), thus demonstrating the advantages of AI in clinical diagnosis.

4.3. Application of artificial intelligence in cancer medical imaging
To date, AI has been utilized in many medical imaging fields such as to CT and magnetic resonance imagery (MRI), and has facilitated accurate diagnosis and treatment. Liu et al. [93] developed a novel DL architecture (XmasNet) based on CNNs for the classification of prostate cancer lesions using 3D multiparametric MRI data provided by the PROSTATEx challenge. Their proposed model outperformed 69 methods among 33 participating groups, and achieved the second-highest AUC value (0.84) in the PROSTATEx challenge. This study showed the great potential of DL for cancer imaging. Wang et al. [94] compared DL with a DCNN and non-DL with scale-invariant feature transform (SIFT) image feature and bag-of-words (BoW) to distinguish patients confirmed to have prostate cancer (PCa) from those with prostate benign conditions (BCs) such as prostatitis or prostate benign hyperplasia (BPH). Their results suggested that DL with the DCNN is superior to non-DL learning with SIFT image feature and BoW for differentiating fully automated PCa patients from BCs patients. These results proved that DL can be extended to image modalities such as MRI, CT, and PET scans of other organs.

Wang et al. [95] devised a novel DL feature and Cox proportional hazard (DL-CPH) regression to extract effective CT-based prognostic biomarkers for high-grade serous ovarian cancer (HGSOC). Their proposed non-invasive and preoperative model could also predict individualized recurrence for HGSOC. Thus, the prognostic analysis method may utilize CT data without follow-up for prognostic biomarker extraction. Medeiros et al. [96] introduced a novel DL approach to assess fundus photographs and provide quantitative information about the amount of neural damage, which can then be used to diagnose and stage glaucoma. In addition, the DL-based algorithm could overcome limitations of human labeling and be applied to other areas of ophthalmology.

5. Challenges and future outlook
AI can successfully handle complex nonlinear relationships, fault tolerance, parallel distributed processing, and learning [97]. Given its advantages of self-adaptation, simultaneous treatment of quantitative and qualitative knowledge, and validated results from a number of clinical studies in multiple fields [98]. AI clearly has varied uses in the field of clinical medicine [99]. It not only makes full use of the various aspects of clinical diversity [100,101], but also helps to address the current lack of objectivity and universality in expert systems [102]. The application of AI can help hospitals train junior physicians in clinical diagnosis and decision-making. A growing number of research papers are reporting about the impressive diagnostic and prognosis performance of computer systems built using ML [103,104]. DL techniques, in particular, are transforming our ability to interpret imaging data [105,106]. These results may improve sensitivity and ensure fewer false positives than radiologists. However, they also run the risk of overfitting the training data, resulting in brittle degraded performance in certain settings [107]. Thus, ML often involves a tradeoff between accuracy and intelligibility. More accurate models, such as boosted trees, random forests, and neural nets, are usually not intelligible, whereas more intelligible models, such as logistic regression, naive-Bayes, and single decision trees, often provide significantly worse accuracy [108]. Recent work using advanced in vivo imaging, computational modelling, and animal modelling has identified barriers in the tumor microenvironment that hinder therapy and promote tumor progression [9]. Along with other risk factors identified from blood counts, red cell distribution width was used in an ML-based approach to generate a clinical data-driven prediction model that was capable of predicting acute myeloid leukemia 6–12 months before diagnosis with high specificity (98.2%) but low sensitivity (25.7%) [111,112]. Thus, while the application of AI in clinical cancer is likely to increase, the following challenges should be met in order for it to remain viable.

AI technology faces some important challenges that must be resolved to ensure its use in cancer diagnosis and prognosis [113]. For example, medical imaging data cannot be used as input data directly. It is crucial to extract features from the imaging data and process them. Development and popularization of technology, in addition, the weights coefficient in the neural network models are tested, calculated, and the confidence interval is reasonable, so medical interpretation need further research [114]. Increasing research on ANNs will likely result in their increased use in the field of clinical medicine. While the importance of AI to this field is recognized, the joint efforts of computer experts and medical experts toward ensuring interdisciplinary personnel training and collaboration are crucial. Only then can the potential of this technology be put to practical and economic application by medical staff [115]. A more pessimistic view was offered in Ref. [116], which referred to inherent uncertainties in medicine, and the possibility that the “black box” of neural networks/ML applications will reduce physician skills and soon transform some sectors of healthcare in ways that may appear to be practical and economic but with unintended negative consequences. Another crucial issue with regard to the future of AI in medicine involves privacy and data security assurances [117]. While recent years have witnessed much enthusiasm about the potential of “big data” and ML-based solutions, to date, only a few examples exist to illustrate the impact of AI on current clinical practice [10,[118], [119], [120]]. Obermeyer et al. [121]showed that attention has to shift to new statistical tools from ML to be critical for anyone practicing medicine in the 21st century. The stimulating debate that whether AI are “smarter” than human practitioners is largely irrelevant, and we will consistently improve our collective health by using every information and data resource [107].

6. Conclusion
AI is slowly pervading all aspects of our lifestyle, especially medicine. The review presented in this paper shows that researchers are rapidly acquiring a much deeper understanding of the challenges and opportunities presented by AI as an intelligent information science in the field of cancer diagnosis and care. The potential of AI for various types of cancer prognosis and diagnosis is reported in this paper. But, the limit of review is that we did not include the genomics and radiomics data applied by AI to acquire clinical precise medicine. We expect that AI-based clinical cancer research will result in a paradigm shift in cancer treatment, thereby resulting in dramatic improvement in patient survival due to enhanced prediction rates. Thus, it is logical to expect that the challenges of cancer prognosis and diagnosis will be solved by advances in AI in the foreseeable future.

CRediT authorship contribution statement
Shigao Huang: Conceptualization, Data curation, Formal analysis, Methodology, Writing – original draft, Writing - review & editing. Jie Yang: Funding acquisition, Formal analysis, Writing – original draft, Writing - review & editing. Simon Fong: Writing - review & editing. Qi Zhao: Writing - review & editing, Funding acquisition.

Declaration of competing interest
The authors declare no competing financial interests.

Acknowledgments
This work was funded by the Science and Technology Development Fund of Macau (FDCT/131/2016/A3, FDCT/0015/2018/A1, FDCT/126/2014/A3) and Start-up Research Grand (SRG2016-00082-FHS), the Multi-Year Research Grant (MYRG2019-00069-FHS, MYRG2016-00069-FST), the intramural research program of Faculty of Health Sciences, University of Macau, Guangzhou Science and Technology Innovation and Development of Special Funds, Grant no. EF003/FST-FSJ/2019/GSTIC, and Grant no. EF004/FST-FSJ/2019/GSTI, key project of Chongqing Industry&Trade Polytechnic (ZR201902,190101), and the project of science and technology research program of Chongqing Education Commission of China(KJQN201903601).

References
[1]
R.L. Siegel, K.D. Miller, A. Jemal
Cancer statistics, 2019
CA A Cancer J. Clin., 69 (2019), pp. 7-34
CrossRefView Record in ScopusGoogle Scholar
[2]
C.P.L. Simmons, D.C. McMillan, K. McWilliams, T.A. Sande, K.C. Fearon, S. Tuck, M.T. Fallon, B.J. Laird
Prognostic tools in patients with advanced cancer: a systematic review
J. Pain Symptom Manag., 53 (5) (2017), pp. 962-970 e910
View Record in ScopusGoogle Scholar
[3]
S. Huang, Y. Dang, F. Li, W. Wei, Y. Ma, S. Qiao, Q. Wang
Biological intensity-modulated radiotherapy plus neoadjuvant chemotherapy for multiple peritoneal metastases of ovarian cancer: a case report
Oncol. Lett. (2015), pp. 1239-1243
CrossRefView Record in ScopusGoogle Scholar
[4]
S. Huang, Q. Zhao
Nanomedicine-combined immunotherapy for cancer
Curr. Med. Chem. (2019), 10.2174/0929867326666190618161610
Google Scholar
[5]
S. Huang, C.I. Fong, M. Xu, B.-n. Han, Z. Yuan, Q. Zhao
Nano-loaded natural killer cells as carriers of indocyanine green for synergetic cancer immunotherapy and phototherapy
J. Innov. Opt. Health Sci., 12 (2019), p. 1941002
CrossRefGoogle Scholar
[6]
Z. Obermeyer, E.J. Emanuel
Predicting the future - big data, machine learning, and clinical medicine
N. Engl. J. Med., 375 (2016), pp. 1216-1219
CrossRefView Record in ScopusGoogle Scholar
[7]
P.E. Kinahan, R.J. Gillies, H. Hricak
Radiomics images are more than pictures, They are data
Radiology, 278 (2015), pp. 563-577
Google Scholar
[8]
A. Allahyar, J. Ubels, J. de Ridder
A data-driven interactome of synergistic genes improves network-based cancer outcome prediction
PLoS Comput. Biol., 15 (2019), Article e1006657
CrossRefGoogle Scholar
[9]
M.J. Mitchell, R.K. Jain, R. Langer
Engineering and physical sciences in oncology: challenges and opportunities
Nat. Rev. Cancer, 17 (2017), pp. 659-675
CrossRefView Record in ScopusGoogle Scholar
[10]
A. Hosny, C. Parmar, J. Quackenbush, L.H. Schwartz, H. Aerts
Artificial intelligence in radiology
Nat. Rev. Cancer, 18 (2018), pp. 500-510
CrossRefView Record in ScopusGoogle Scholar
[11]
R.C. Deo
Machine learning in medicine
Circulation, 132 (2015), pp. 1920-1930
View Record in ScopusGoogle Scholar
[12]
S. Jha, E.J. Topol
Adapting to artificial intelligence: radiologists and pathologists as information specialists
J. Am. Med. Assoc., 316 (2016), pp. 2353-2354
CrossRefView Record in ScopusGoogle Scholar
[13]
D. Wong, S. Yip
Machine learning classifies cancer
Nature, 555 (2018), pp. 446-447
CrossRefView Record in ScopusGoogle Scholar
[14]
P. Glare, C. Sinclair, M. Downing, P. Stone, M. Maltoni, A. Vigano
Predicting survival in patients with advanced disease
Eur. J. Cancer, 44 (2008), pp. 1146-1156
ArticleDownload PDFView Record in ScopusGoogle Scholar
[15]
C.P.L. Simmons, D.C. McMillan, K. McWilliams, T.A. Sande, K.C. Fearon, S. Tuck, M.T. Fallon, B.J. Laird
Prognostic tools in patients with advanced cancer: a systematic review
J. Pain Symptom Manag., 53 (2017), pp. 962-970 e910
View Record in ScopusGoogle Scholar
[16]
K. Kourou, T.P. Exarchos, K.P. Exarchos, M.V. Karamouzis, D.I. Fotiadis
Machine learning applications in cancer prognosis and prediction
Comput. Struct. Biotechnol. J., 13 (2015), pp. 8-17
ArticleDownload PDFView Record in ScopusGoogle Scholar
[17]
H.M. Zolbanin, D. Delen, A. Hassan Zadeh
Predicting overall survivability in comorbidity of cancers: a data mining approach
Decis. Support Syst., 74 (2015), pp. 150-161
ArticleDownload PDFView Record in ScopusGoogle Scholar
[18]
D. Chen, K. Xing, D. Henson, L. Sheng, A.M. Schwartz, X. Cheng
Developing prognostic systems of cancer patients by ensemble clustering
J. Biomed. Biotechnol., 2009 (2009), p. 632786
Google Scholar
[19]
C. Denkert, G. von Minckwitz, S. Darb-Esfahani, B. Lederer, B.I. Heppner, K.E. Weber, J. Budczies, J. Huober, F. Klauschen, J. Furlanetto, W.D. Schmitt, J.-U. Blohmer, T. Karn, B.M. Pfitzner, S. Kümmel, K. Engels, A. Schneeweiss, A. Hartmann, A. Noske, P.A. Fasching, C. Jackisch, M. van Mackelenbergh, P. Sinn, C. Schem, C. Hanusch, M. Untch, S. Loibl
Tumour-infiltrating lymphocytes and prognosis in different subtypes of breast cancer: a pooled analysis of 3771 patients treated with neoadjuvant therapy
Lancet Oncol., 19 (2018), pp. 40-50
ArticleDownload PDFView Record in ScopusGoogle Scholar
[20]
Y. Mintz, R. Brodie
Introduction to artificial intelligence in medicine
Minim. Invasive Ther. Allied Technol. (2019), pp. 1-9
View Record in ScopusGoogle Scholar
[21]
Z. Qian, Y. Li, Y. Wang, L. Li, R. Li, K. Wang, S. Li, K. Tang, C. Zhang, X. Fan, B. Chen, W. Li
Differentiation of glioblastoma from solitary brain metastases using radiomic machine-learning classifiers
Cancer Lett., 451 (2019), pp. 128-135
ArticleDownload PDFView Record in ScopusGoogle Scholar
[22]
A. Tan, H. Huang, P. Zhang, S. Li
Network-based cancer precision medicine: a new emerging paradigm
Cancer Lett., 458 (2019), pp. 39-45
ArticleDownload PDFView Record in ScopusGoogle Scholar
[23]
S. Huang, J. Yang, S. Fong, Q. Zhao
Mining prognosis index of brain metastases using artificial intelligence
Cancers, 11 (2019), p. 1140
CrossRefGoogle Scholar
[24]
Y. Dang, X. Li, Y. Ma, X. Li, T. Yang, W. Lu, S. Huang
18F-FDG-PET/CT-guided intensity-modulated radiotherapy for 42 FIGO III/IV ovarian cancer: a retrospective study
Oncol. Lett., 17 (2019), pp. 149-158
View Record in ScopusGoogle Scholar
[25]
H.S. Gao HX, J.F. Du, X.C. Zhang, N. Jiang, W.X. Kang, J. Mao, Q. Zhao
Comparison of prognostic indices in NSCLC patients with brain metastases after radiosurgery
Int. J. Biol. Sci., 14 (2018), pp. 2065-2072
Google Scholar
[26]
A. Enshaei, C.N. Robson, R.J. Edmondson
Artificial intelligence systems as prognostic and predictive tools in ovarian cancer
Ann. Surg. Oncol., 22 (2015), pp. 3970-3975
CrossRefView Record in ScopusGoogle Scholar
[27]
S.H. Khan U, J.P. Choi, et al.
wFDT weighted fuzzy decision trees for prognosis of breast cancer survivability
Proceedings of the 7th Australasian Data Mining Conference, Australian Computer Society (2008), pp. 141-152
Google Scholar
[28]
S. Jhajharia, H.K. Varshney, S. Verma, R. Kumar
A neural network based breast cancer prognosis model with PCA processed features
B.U. Department of Computer Engineering, Jaipur, India 304022
2016 International Conference on Advances in Computing, Communications and Informatics (ICACCI), IEEE, Jaipur, India (2016), pp. 1896-1901
CrossRefView Record in ScopusGoogle Scholar
[29]
T. Ching, X. Zhu, L.X. Garmire
Cox-nnet: an artificial neural network method for prognosis prediction of high-throughput omics data
PLoS Comput. Biol., 14 (2018), Article e1006076
CrossRefGoogle Scholar
[30]
A. Bomane, A. Gonçalves, P.J. Ballester
Paclitaxel response can Be predicted with interpretable multi-variate classifiers exploiting DNA-methylation and miRNA data
Front. Genet., 10 (2019)
Google Scholar
[31]
D. Sun, M. Wang, A. Li
A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data
IEEE ACM Trans. Comput. Biol. Bioinform, 16 (3) (2018), pp. 841-850
CrossRefView Record in ScopusGoogle Scholar
[32]
C.C. L, S.W. N, W.W. H
Application of artificial neural network-based survival analysis on two breast cancer datasets
AMIA Annual Symposium Proceedings, American Medical Informatics Association (2007), p. 130
Google Scholar
[33]
K. Park, A. Ali, D. Kim, Y. An, M. Kim, H. Shin
Robust predictive model for evaluating breast cancer survivability
Eng. Appl. Artif. Intell., 26 (2013), pp. 2194-2205
ArticleDownload PDFView Record in ScopusGoogle Scholar
[34]
D. Delen, G. Walker, A. Kadam
Predicting breast cancer survivability: a comparison of three data mining methods
Artif. Intell. Med., 34 (2005), pp. 113-127
ArticleDownload PDFView Record in ScopusGoogle Scholar
[35]
Y. Sun, S. Goodison, J. Li, L. Liu, W. Farmerie
Improved breast cancer prognosis through the combination of clinical and genetic markers
Bioinformatics, 23 (2007), pp. 30-37
CrossRefView Record in ScopusGoogle Scholar
[36]
O. Gevaert, F. De Smet, D. Timmerman, Y. Moreau, B. De Moor
Predicting the prognosis of breast cancer by integrating clinical and microarray data with Bayesian networks
Bioinformatics, 22 (2006), pp. e184-190
CrossRefView Record in ScopusGoogle Scholar
[37]
R.J. Kuo, M.H. Huang, W.C. Cheng, C.C. Lin, Y.H. Wu
Application of a two-stage fuzzy neural network to a prostate cancer prognosis system
Artif. Intell. Med., 63 (2015), pp. 119-133
ArticleDownload PDFView Record in ScopusGoogle Scholar
[38]
S. Zhang, Y. Xu, X. Hui, F. Yang, Y. Hu, J. Shao, H. Liang, Y. Wang
Improvement in prediction of prostate cancer prognosis with somatic mutational signatures
J. Cancer, 8 (2017), pp. 3261-3267
CrossRefView Record in ScopusGoogle Scholar
[39]
H. Lu, H. Wang, S.W. Yoon
A dynamic gradient boosting machine using genetic optimizer for practical breast cancer prognosis
Expert Syst. Appl., 116 (2019), pp. 340-350
ArticleDownload PDFView Record in ScopusGoogle Scholar
[40]
P. Vasudevan, T. Murugesan
Cancer subtype discovery using prognosis-enhanced neural network classifier in multigenomic data
Technol. Cancer Res. Treat., 17 (2018)
1533033818790509
Google Scholar
[41]
D.W. Tian, Z.L. Wu, L.M. Jiang, J. Gao, C.L. Wu, H.L. Hu
Neural precursor cell expressed, developmentally downregulated 8 promotes tumor progression and predicts poor prognosis of patients with bladder cancer
Cancer Sci., 110 (2019), pp. 458-467
CrossRefView Record in ScopusGoogle Scholar
[42]
Z. Hasnain, J. Mason, K. Gill, G. Miranda, I.S. Gill, P. Kuhn, P.K. Newton
Machine learning models for predicting post-cystectomy recurrence and survival in bladder cancer patients
PLoS One, 14 (2019), Article e0210976
CrossRefGoogle Scholar
[43]
H.E. Biglarian A, A. Kazemnejad, et al.
Application of artificial neural network in predicting the survival rate of gastric cancer patients
Iran. J. Public Health, 40 (2011)
Google Scholar
[44]
L. Zhu, W. Luo, M. Su, H. Wei, J. Wei, X. Zhang, C. Zou
Comparison between artificial neural network and Cox regression model in predicting the survival rate of gastric cancer patients
Biomed. Rep., 1 (2013), pp. 757-760
CrossRefView Record in ScopusGoogle Scholar
[45]
L. Bottaci, P.J. Drew, J.E. Hartley, M.B. Hadfield, R. Farouk, P.W.R. Lee, I.M.C. Macintyre, G.S. Duthie, J.R.T. Monson
Artificial neural networks applied to outcome prediction for colorectal cancer patients in separate institutions
The Lancet, 350 (1997), pp. 469-472
