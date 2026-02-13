%Version 3 December 2023
% See section 11 of the User Manual for version history
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                 %%
%% Please do not use \input{...} to include other tex files.       %%
%% Submit your LaTeX manuscript as one .tex document.              %%
%%                                                                 %%
%% All additional figures and files should be attached             %%
%% separately and not embedded in the \TeX\ document itself.       %%
%%                                                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%\documentclass[referee,sn-basic]{sn-jnl}% referee option is meant for double line spacing

%%=======================================================%%
%% to print line numbers in the margin use lineno option %%
%%=======================================================%%

%%\documentclass[lineno,sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style

%%======================================================%%
%% to compile with pdflatex/xelatex use pdflatex option %%
%%======================================================%%

%%\documentclass[pdflatex,sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style


%%Note: the following reference styles support Namedate and Numbered referencing. By default the style follows the most common style. To switch between the options you can add or remove Numbered in the optional parenthesis. 
%%The option is available for: sn-basic.bst, sn-vancouver.bst, sn-chicago.bst%  
 
%%\documentclass[pdflatex,sn-nature]{sn-jnl}% Style for submissions to Nature Portfolio journals
%%\documentclass[pdflatex,sn-basic]{sn-jnl}% Basic Springer Nature Reference Style/Chemistry Reference Style


\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}% Math and Physical Sciences Numbered Reference Style 

% TODO: uncomment the line above, the line below is for double-spacing only
% \documentclass[referee,pdflatex,sn-mathphys-num]{sn-jnl}


%%\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}% Math and Physical Sciences Author Year Reference Style
%%\documentclass[pdflatex,sn-aps]{sn-jnl}% American Physical Society (APS) Reference Style
%%\documentclass[pdflatex,sn-vancouver,Numbered]{sn-jnl}% Vancouver Reference Style
%%\documentclass[pdflatex,sn-apa]{sn-jnl}% APA Reference Style 
%%\documentclass[pdflatex,sn-chicago]{sn-jnl}% Chicago-based Humanities Reference Style

%%%% Standard Packages
%%<additional latex packages if required can be included here>

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathrsfs}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%
\usepackage{algorithm}%
\usepackage{algorithmicx}%
\usepackage{algpseudocode}%
\usepackage{listings}%
\usepackage{hyperref}%
\usepackage{enumitem}
%%%%

%%%%%=============================================================================%%%%
%%%%  Remarks: This template is provided to aid authors with the preparation
%%%%  of original research articles intended for submission to journals published 
%%%%  by Springer Nature. The guidance has been prepared in partnership with 
%%%%  production teams to conform to Springer Nature technical requirements. 
%%%%  Editorial and presentation requirements differ among journal portfolios and 
%%%%  research disciplines. You may find sections in this template are irrelevant 
%%%%  to your work and are empowered to omit any such section if allowed by the 
%%%%  journal you intend to submit to. The submission guidelines and policies 
%%%%  of the journal take precedence. A detailed User Manual is available in the 
%%%%  template package for technical guidance.
%%%%%=============================================================================%%%%

%% as per the requirement new theorem styles can be included as shown below
\theoremstyle{thmstyleone}%
\newtheorem{theorem}{Theorem}%  meant for continuous numbers
%%\newtheorem{theorem}{Theorem}[section]% meant for sectionwise numbers
%% optional argument [theorem] produces theorem numbering sequence instead of independent numbers for Proposition
\newtheorem{proposition}[theorem]{Proposition}% 
%%\newtheorem{proposition}{Proposition}% to get separate numbers for theorem and proposition etc.

\theoremstyle{thmstyletwo}%
\newtheorem{example}{Example}%
\newtheorem{remark}{Remark}%

\theoremstyle{thmstylethree}%
\newtheorem{definition}{Definition}%

\raggedbottom
%%\unnumbered% uncomment this for unnumbered level heads

\begin{document}

\title[Article Title]{Noise-Robust QSAR: Evaluating Probabilistic Methods Across Molecular Representations}
%%=============================================================%%
%% GivenName	-> \fnm{Joergen W.}
%% Particle	-> \spfx{van der} -> surname prefix
%% FamilyName	-> \sur{Ploeg}
%% Suffix	-> \sfx{IV}
%% \author*[1,2]{\fnm{Joergen W.} \spfx{van der} \sur{Ploeg} 
%%  \sfx{IV}}\email{iauthor@gmail.com}
%%=============================================================%%

\author[1]{\fnm{Adelaide} \sur{Punt}}\email{adelaidepunt@gmail.com}

\author[2]{\fnm{Thierry} \sur{Hanser}}

\author[2]{\fnm{Stephane} \sur{Werner}}

\author*[1]{\fnm{Garrett} \sur{Morris}}\email{garrett.morris@stats.ox.ac.uk}

\affil*[1]{\orgdiv{Department of Statistics}, \orgname{University of Oxford}, \orgaddress{\street{24-29 St Giles'}, \city{Oxford}, \postcode{OX1 3LB}, \country{UK}}}

\affil[2]{\orgdiv{Lhasa Limited}, \orgname{Granary Wharf House}, \orgaddress{\street{2 Canal Wharf}, \city{Leeds}, \postcode{LS11 5PS}, \country{UK}}}

%%==================================%%
%% Sample for unstructured abstract %%
%%==================================%%
\abstract{Predictive models for chemical bioactivity are limited by experimental noise and intrinsic biological and chemical variability. Traditional machine learning approaches treat these noisy experimental labels as ground truth and can overfit to the noise itself. Probabilistic modeling, which provides uncertainty quantification, has been used to address some of these issues; however, the evaluation of different probabilistic approaches in terms of comparative effectiveness across different noise conditions and molecular representations remains limited. In this study, we examine noise robustness and uncertainty quantification in a wide range of contexts, comparing the effects of different molecular representations and model architectures under a variety of types of label noise. We injected artificial noise into both clean quantum-chemical datasets and experimental ADME datasets. We demonstrate that while molecular representation explains more variance in overall QSAR performance, model architecture explains more variance in specifically noise robustness. The modeling approaches that showed the highest noise robustness were NGBoost, SVM, and full Bayesian neural networks. Full Bayesian transformation significantly improved NN robustness, while last-layer transformation did not. Among probabilistic models, NGBoost provides the most useful uncertainty estimates under noisy conditions. Model rankings remain highly stable across different types of artificial noise ($W = 0.957$) but do not transfer to external datasets, indicating that dataset-specific factors dominate robustness on smaller experimental datasets. We introduce NoiseInject, an open-source benchmarking framework for evaluating noise robustness across arbitrary datasets, optimized for molecular data but usable on other subjects.}


%%================================%%
%% Sample for structured abstract %%
%%================================%%

% \abstract{\textbf{Purpose:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Methods:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Results:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.
% 
% \textbf{Conclusion:} The abstract serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. The abstract must not include subheadings (unless expressly permitted in the journal's Instructions to Authors), equations or citations. As a guide the abstract should not exceed 200 words. Most journals do not set a hard limit however authors are advised to check the author instructions for the journal they are submitting to.}

\keywords{Bayesian, QSAR, uncertainty quantification, label noise, molecular representations, conformal prediction}

%%\pacs[JEL Classification]{D8, H51}

%%\pacs[MSC Classification]{35A01, 65L10, 65L12, 65L20, 65L70}

\maketitle


\section*{Abbreviations}

\begin{description}[style=unboxed,leftmargin=0cm]
    \item[ADME] Absorption, distribution, metabolism, and excretion
    \item[BNN] Bayesian neural network
    \item[DFT] Density functional theory
    \item[ECFP4] Extended connectivity fingerprint with radius 4
    \item[ECE] Expected calibration error
    \item[GP] Gaussian process
    \item[HLM] Human liver microsomal stability
    \item[LightGBM] Light Gradient Boosting Machine
    \item[LNL] Learning with noisy labels
    \item[NDS] Noise degradation slope
    \item[NGBoost] Natural gradient boosting
    \item[PDV] Physicochemical descriptor vector
    \item[PLS] Partial least squares
    \item[QRF] Quantile regression forest
    \item[QSAR] Quantitative structure-activity relationship
    \item[QSPR] Quantitative structure-property relationship
    \item[RF] Random forest
    \item[RLM] Rat liver microsomal stability
    \item[SMILES] Simplified molecular input line entry system
    \item[SNS] Sort \& Slice
    \item[SVM] Support vector machine
\end{description}

\clearpage

\section{Introduction}\label{sec1}

New avenues in the drug discovery pipeline have opened thanks to advances in large data collection, storage, and analysis. Therapeutic targets are identified, potential drug candidates are screened and optimized, and the most promising of those go on to pre-clinical and clinical trials \citep{Lill2007}. There are various methods by which potential drug candidates can be screened. Virtual screening, which allows for the rapid and large-scale analysis of libraries of data, is a particularly useful approach used to speed up the drug discovery process. \citep{Neves2018}. One standard screening method is the prediction of chemical bioactivities and physical properties of such candidates. Quantitative structure-activity relationship (QSAR) models can rapidly make these predictions, act as a filter for compounds with undesirable properties, and thus reduce the time and cost associated with experimental screening \citep{Sabando2021}. Specifically, QSAR models predict the relationship between molecular features encoded within a particular molecular representation and a target property \citep{Sabando2021}. A QSAR model typically consists of a molecular representation paired with a machine learning (ML) model. An ideal molecular representation will allow a model to learn only the relevant information about each molecule with respect to the target property. Molecular featurization refers to the process of converting a molecule into a computational representation, or an embedding \citep{Sabando2021}. 

QSAR models are sensitive to the quality of the input data on which they are trained, in the form of experimental noise from measurement errors, variability under experimental conditions, provenance of the data and intrinsic biological and chemical variability. \cite{Kolmar2021} show that the variance associated with the experimental data can contribute more to the prediction error than the model's error. Even measures of model performance may be flawed if test and validation sets are laden with noise. In the context of activity prediction, the inherent variability within experimental setups and assays, batch effect, measurement error, and biological variability introduce additional layers of complexity \citep{Kolmar2021, Fourches2010}. All of this increases the likelihood of encountering noise in the data \citep{Fang2022}. Even the representations themselves may be misleading, as they encode structural information and may not account for all the potential conformations a molecule can adopt, primarily when binding to a particular target. 

Numerous studies have explored learning with noisy labels (LNL) in the broader context of ML and specifically within QSAR models. \cite{Song2022} provides a comprehensive review of deep LNL, identifying five main approaches: modifying architectures to better handle noise in training, employing regularization to reduce overfitting to noise, adjusting loss functions to be less sensitive to outliers or label flips, post-processing to adjust the weights for noisy data, and sample selection to select presumably clean data to pre-train. Although some QSAR models can overcome noise, predictive performance depends on the algorithm. \cite{Cortes2015} examined 12 different algorithms across 12 datasets and 10 noise levels, finding that no single algorithm was overall best at noise; certain algorithms perform better or worse at different levels of noise. This is a recurring theme in LNL research: There is no universal solution to handling noise in cheminformatics data. This is consistent with the No Free Lunch Theorem, which states that there is no single learning algorithm that can outperform all others in all data distributions \citep{Wolpert1997}. This suggests that there is no single best model or molecular representation for noise robustness; it depends on the underlying data. Certain techniques can be more effective with different types of noise. For example, small-loss selection works particularly well for symmetric noise \citep{Song2022}. Some techniques that may be robust to experimental noise may not be fit for the particular QSAR task at hand. Combinations of LNL approaches have yielded promising results, specifically sample selection combined with semi-supervised learning on rejected data, presumably noisy samples. Many LNL techniques were developed for image classification tasks. The data properties of molecular datasets differ significantly, so adjustments are needed to adapt some of these techniques \citep{Song2022}.

% TODO [REPETITION]: The next 5 paragraphs (labels as distributions, Bayesian methods,
% GPs, trees, NNs/BNNs) are repeated almost verbatim in Section 2.3 (Models).
% RECOMMENDATION: Trim these intro paragraphs to 1-2 motivating sentences each,
% and keep the detailed descriptions in Methods only. The intro should motivate
% *why* we study these approaches, not *how* they work.
One key issue with QSAR models is the standard ML assumption that labels represent true values, as most chemical data sets lack sufficient replications to provide a robust statistical representation of the underlying distributions \citep{Kolmar2021}. A single measurement, or even multiple measurements for a given label, will not guarantee that this label accurately reflects the population mean \citep{Kolmar2021}. Most ML models rely on the assumption that the training data is, in fact, an accurate reflection of the population. Therefore, they treat these potentially noisy labels as discrete quantities rather than distributions, which opens the door to overfitting to noise \citep{Kolmar2021}. However, probabilistic methods such as Gaussian processes (GPs) and Bayesian neural networks (BNNs) do not make this assumption \citep{Kolmar2021}.

Unlike standard deterministic ML methods, Bayesian methods model distributions over predictions using Bayes' rule, allowing for the quantification of uncertainty within parameters and labels. Bayesian methods treat the model parameters as random variables with distributions. Uncertainty can also be estimated from other non-deterministic methods, including ensemble \citep{lakshminarayanan2017}, bootstrap modeling \citep{palmer2022}, and federated learning \citep{Hanser2023}. The uncertainty values derived from these non-deterministic methods can be further decomposed into epistemic and aleatoric components. Epistemic uncertainty accounts for uncertainty within the model and can be minimized with sufficient data. Aleatoric uncertainty refers to the noise inherent in the data itself, meaning more data will not improve aleatoric uncertainty \citep{Hllermeier2021, kendall2017}.

One ML method that has had particular success in molecular property prediction is Gaussian Processes (GPs) \citep{Obrezanova2007, gauche}. A GP is a non-parametric model that outputs a Gaussian predictive distribution over every data point, providing an uncertainty estimate for each prediction \citep{Obrezanova2007}. GPs have been known to match or exceed the predictive performance of traditional QSAR models such as support vector machines (SVMs) or partial least squares (PLS), while also providing insight into the reliability of those predictions \citep{Obrezanova2007}. The GP defines a distribution over functions, whose structure is determined by the kernel. These kernels can be modified to fit specific domains or functionalities. Typically, a Tanimoto kernel is used when working with molecular fingerprints \citep{Ralaivola2005, moss2020, gauche}. One of the major limitations to using GPs in practice is computational scaling. Inference has $O(N^3)$ time complexity and $O(N^2)$ memory, where $N$ is the number of training points \citep{Rasmussen2005}. This can be mitigated with sparse GP approximations, but these introduce approximation error and can be more difficult to tune \citep{quinonero2005}. Ensembles of trees or BNNs are typically more efficient for larger data sizes \citep{lakshminarayanan2017}.

Individual decision trees tend to have high variance and often overfit, particularly at large depths \citep{Breiman1984}. Small modifications made to the training data can result in large differences in the structure of the tree, resulting in unstable predictions \citep{Breiman2001}. Ensemble methods are used to counteract this instability, aggregating predictions from many trees. Random forests (RFs) aggregate predictions from trees trained on bootstrap samples of data with random subsets of features at each split \citep{Breiman2001}. Gradient boosting methods such as XGBoost sequentially fit trees to the residuals of previous iterations, progressively reducing the prediction error \citep{Chen2016}. Bootstrapping ensures that each tree is trained on a slightly different subset of data, which effectively reduces the influence of any individual sample \citep{Breiman2001}. This is especially beneficial with noisy samples. The selection of random features at every node decorrelates the trees, preventing systematic biases from propagating through the ensemble \citep{Breiman2001}. The aggregation of trees further prevents overfitting to noise by smoothing out prediction errors from individual noisy labels. XGBoost also has a strong track record in molecular property prediction tasks \citep{Mustapha2016, Tian2022}. Alongside XGBoost, we used Natural Gradient Boosting (NGBoost) \citep{Duan2020} that extends traditional gradient-boosted methods by outputting predictive distributions rather than point estimates. NGBoost treats the parameters of a chosen parametric distribution as regression targets and learns them via boosting with a natural gradient update rule \citep{Duan2020}. Although NGBoost is model-agnostic, it is typically used with decision trees \citep{Duan2020}.

NNs are a popular choice in QSAR research, though they are often outperformed by more traditional methods \citep{Baskin2008, Koutsoukas2017}. Decision trees and related ensemble models tend to perform stronger on smaller datasets, overfit less, and don't rely on a static 3D molecular structure like NNs do \citep{Baskin2008, Koutsoukas2017}. NNs are often deterministic; however, they can be transformed into probabilistic models through several approaches. BNNs introduce priors on the weights and compute a posterior distribution on those weights given the input data, providing uncertainty quantification. This can be done by approximations, including Monte Carlo dropout, variational inference, or ensembles \citep{gal2016}. To improve computational efficiency, several strategies exist for converting NNs to BNNs. One straightforward yet computationally expensive approach is to replace all linear layers with Bayesian layers (full Bayesian transformation, or full-BNN). Replacing only the final layer is computationally lighter (last-layer transformation, or last-layer-BNN). One particularly efficient approach is Variational Bayesian Last Layers, which use a variational formulation that can be trained with only quadratic complexity in the last layer width \citep{Harrison2024} (variational transformation, or var-BNN). Bayesian approaches also act as forms of regularization \citep{Burden2009}, as dropouts help to prevent overfitting.

% TODO: fix the deng reference (possessive form with \cite)
Although numerous studies have explored the effects of noise on ML algorithms for molecular property prediction, comparisons between different molecular representations remain limited. \cite{Kolmar2021} examined the predictive limits of QSAR in the presence of noise, showing that by learning the underlying trends rather than fitting to noise, the models can achieve predictions more accurate than those of their training data. They assumed experimental noise could be modeled as Gaussian-distributed, using the Central Limit Theorem to argue that aggregated experimental measurements from multiple independent sources of error exhibit a Gaussian distribution. However, \cite{Heid2023} used mean-variance estimation and bias-variance decomposition to show that noise can be tied to specific modalities, for example structure dependence. As an example, nitrogen-containing functional groups introduce measurement deviations, implying the presence of systematic noise. The choice of molecular representation also impacts the predictive performance of QSAR models. \cite{Deng2023} conducted a large-scale evaluation of 62,820 models and found that Extended Connectivity Fingerprint (ECFP) fingerprints frequently outperform learned graph neural network (GNN) representations, with dataset size proving more important than representation choice.

This research addresses three main questions concerning the robustness of molecular machine learning models under label noise. First, we investigate the contributions of molecular representation and model architecture to both overall prediction performance and, more specifically, noise robustness. Second, we compare how noise-robust probabilistic models are to their deterministic counterparts, and whether uncertainty estimates correlate with prediction error or label noise under noisy conditions. Third, we assess the generalizability of noise-robustness patterns across different noise-injection mechanisms and molecular properties. Together, these questions aim to identify what, if anything, makes a QSAR model robust to noise. Finally, we have released NoiseInject, an open-source Python package that acts as a noise benchmarking tool. It allows the user to add noise in different scenarios to an arbitrary dataset and to compute metrics to analyze its effects.

\section{Methods}\label{sec2}

\subsection{Dataset}
The majority of experiments were conducted on the QM9 molecular property dataset \citep{Ramakrishnan2014}, which contains approximately 130,000 small organic molecules with pre-computed quantum-mechanical properties obtained from density functional theory (DFT) calculations at the B3LYP/6-31G(2df,p) level of theory \citep{Ramakrishnan2014}. Each machine learning experiment was replicated 10 times with a distinct random seed and a subset of $N = 10{,}000$ molecules. Experiments that involved tracking uncertainty values were only run once due to computational limitations. For each experimental replicate, a new scaffold-based train/validation/test split (80/10/10) implemented by DeepChem \citep{deepchem} was generated from this fixed subset. Scaffold splits were used to prevent structural overlap between sets. When switching from random splitting to a more challenging scenarios such as scaffold splitting, QSAR models typically see performance drops \citep{Heid2023}.

% TODO: modify with new results
We selected HOMO-LUMO energy gap as the primary prediction target in QM9, as it captures electronic characteristics relevant to molecular reactivity and stability \citep{Islam2019, Hllermeier2021}. The HOMO-LUMO gap is a well-established target for quantum chemistry benchmarks as it is defined as the difference between highest occupied and lowest unoccupied molecular orbital energies—directly, meaning it relates to a molecule's electronic excitability, charge transfer capability, and chemical stability \citep{Fediai2023}. We also assumed that thanks to the consistent calculation method, and notwithstanding approximations in the level of theory used, the QM data were "free" of noise.

% TODO [DATA PENDING]: Confirm final dataset sizes and add citations for OpenADMET datasets.
To assess whether noise robustness results generalize beyond the HOMO-LUMO gap, we evaluated model configurations across three additional molecular property datasets with experimentally measured endpoints. From the OpenADMET benchmark suite, we used LogD (lipophilicity) and Caco-2 efflux permeability, both of which are key ADME properties in drug discovery. From ChEMBL, we used hERG-Ki, measuring binding affinity to the cardiac ion channel hERG (Kv11.1) as a continuous regression target (Ki values) rather than a binary classification threshold. All three datasets were evaluated using all six noise injection strategies and the same noise levels ($\sigma \in \{0, 0.1, \ldots, 1.0\}$) as the QM9 experiments, enabling direct comparison of robustness rankings across prediction targets.

SMILES strings were sanitized and canonicalized using \texttt{RDKit}, and target values were  normalized to zero mean and unit variance. Data loading and pre-processing were performed under a global random seed to ensure reproducibility. Prior to modeling, all molecular structures were sanitized and canonicalized using \texttt{RDKit} to remove invalid valence states and standardize atom and bond typing \citep{rdkit}. Target values were mean-centered and normalized to zero mean and unit variance. Data loading and pre-processing were performed with a global random seed to ensure reproducibility.

\subsection{Molecular Representations}

In this study, we evaluated a variety of molecular representations. We used two circular topological fingerprints: the Extended Connectivity Fingerprint with radius 4 (ECFP4) and the \emph{Sort \& Slice} (SNS) fingerprint \citep{Dablander2024}. ECFP4 fingerprints were generated with \texttt{RDKit} \citep{rdkit} using radius $r=2$ and length $d=2048$. Each bit of the fingerprint encodes the hashed identity of one or more circular substructures. One downside of hashed fingerprints is bit collisions, in which chemically distinct substructures map to the same index, resulting effectively in representation-level noise. To address this issue, the SNS fingerprint uses a collision-free alternative to pool ECFP substructures \citep{Dablander2024}. Substructures are sorted by prevalence in the training set, and the top $L$ ($L=1024$) are sliced to create a binary vector. We also used 200-dimensional physicochemical descriptor vectors (PDVs) computed from RDKit molecular descriptors, and one-hot-encoded (OHE) SMILES strings \citep{Bertoni2020}, generated with \texttt{RDKit} \citep{rdkit}. We included mol2vec embeddings \citep{jaeger2018mol2vec}, which apply a Word2Vec-inspired approach to molecular substructures. Molecules are decomposed into Morgan substructures at multiple radii, treated as ``words'' in a molecular ``sentence,'' and a pre-trained Word2Vec model produces 300-dimensional embeddings for each substructure. The molecular embedding is the mean of its constituent substructure vectors, yielding a fixed-length, continuous representation that captures substructural context without requiring task-specific training. Finally, we included graph embeddings generated by Molecular Hypergraph Grammar GNNs (MHG-GNN) \citep{kishimoto2023}. MHG-GNNs are a GIN-based autoencoder pretrained on 1.34 million PubChem molecules using $\beta$-VAE loss, producing 1024-dimensional embeddings through iterative message passing. MHG-GNN was originally developed for materials science, although it has demonstrated strong performance on property prediction tasks for polymers, photoresists, and chromophores \citep{kishimoto2023}. 

All representations are precomputed through a Rust–Python pipeline. Rust performs various heavy-computation tasks involved in feature extraction, normalization, and serialization, and outputs the results into memory-mapped binaries (\texttt{.mmap}) for zero-copy reads. Model training, testing, and validation are then done in Python, reading in from the memory-mapped files.

\subsection{Models}
% TODO [REPETITION]: The paragraph below (labels as distributions, probabilistic vs deterministic)
% repeats material from the Introduction. RECOMMENDATION: Keep here in Methods; trim the intro version.
In this work, we used a wide range of model architectures to understand both deterministic and probabilistic learning behaviors across molecular representations and levels of noise. One key issue with QSAR models is the standard ML assumption that labels represent true values, as most chemical data sets lack sufficient replications to provide a robust statistical representation of the underlying distributions \citep{Kolmar2021}. A single measurement, or even multiple measurements for a given label, will not guarantee that this label accurately reflects the population mean \citep{Kolmar2021}. Most ML models rely on the assumption that the training data is, in fact, an accurate reflection of the population. Therefore, they treat these potentially noisy labels as discrete quantities rather than distributions, which opens the door to overfitting to noise and giving unduly optimistic assessments of the models. However, probabilistic methods such as Gaussian processes (GPs) and Bayesian neural networks (BNNs) do not make this assumption \citep{Kolmar2021}.

Across all models, hyperparameter tuning was performed using Optuna \citep{akiba2019optuna} Bayesian optimization (Tree-structured Parzen Estimator). The tuned hyperparameters were compared against conventional default values for each architecture, and the configuration with better validation performance was retained as the fixed parameter set for all subsequent experiments (Supplementary Table~\ref{tab:supp_s1_hyperparameters}).

% TODO [REPETITION]: Epistemic/aleatoric decomposition paragraph below also appears in
% the Introduction. RECOMMENDATION: Keep here; remove from intro.
The uncertainty values derived from these non-deterministic methods can be further decomposed into epistemic and aleatoric components. Epistemic uncertainty accounts for uncertainty within the model and can be explained away with sufficient data. Aleatoric uncertainty refers to the noise inherent in the data itself, meaning more data will not improve aleatoric uncertainty \citep{Hllermeier2021, kendall2017}.

We included the following variants of decision trees: Random Forest (RF), Quantile Regression Forest (QRF), eXtreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), and Natural Gradient Boosting (NGBoost). RFs are one of the most common choices for QSAR modeling thanks to their robustness and interpretability \citep{Svetnik2003}. However, variance across tree predictions can be used as an uncertainty estimate, and various QSAR studies have linked this ensemble variance to the prediction error; it is not as robust a quantification of uncertainty as found in probabilistic modeling  \citep{Svetnik2003}.  QRFs extend these RF predictions to full distributions \citep{Meinshausen2006}. QRFs keep the distribution of training labels in each leaf and, when they make a prediction, they use all the training instances in the leaves that the new instance falls into, across all trees, to compute quantiles from that set of values. This results in a non-parametric estimate of the conditional distribution of the output \citep{Meinshausen2006}. It inherently accounts for heteroscedasticity, such that if the target variable has more spread in its domain, the quantiles will reflect that. QRF has also been applied in QSAR, though not as widely as RF \citep{Venkatraman2021}. We included XGBoost, a standard among molecular property prediction tasks \citep{Mustapha2016, Tian2022}. We also included LightGBM \citep{ke2017lightgbm}, a gradient boosting framework that uses histogram-based split finding and leaf-wise tree growth for efficient training. LightGBM employs Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to handle high-dimensional features efficiently \citep{ke2017lightgbm}. Alongside XGBoost and LightGBM, we used Natural Gradient Boosting (NGBoost) \citep{Duan2020} that extends traditional gradient-boosted methods by outputting predictive distributions rather than point estimates. NGBoost treats the parameters of a chosen parametric distribution as regression targets and learns them via boosting with a natural gradient update rule \citep{Duan2020}.

We also included Support Vector Machines (SVMs) with radial basis function (RBF) kernels \citep{Vapnik1995}. SVMs are a well-established baseline in QSAR modeling, mapping inputs into a high-dimensional feature space where a maximum-margin hyperplane separates predictions \citep{Svetnik2003}. Although SVMs produce only point predictions without native uncertainty quantification, they serve as a deterministic reference for comparison with probabilistic approaches.

% TODO [REPETITION]: The GP paragraph below overlaps with the GP paragraph in the Introduction.
% RECOMMENDATION: Keep the detailed version here; trim the intro to a brief motivation.
One ML method that has had particular success in molecular property prediction are Gaussian Processes (GPs) \citep{Obrezanova2007, gauche}. A GP is a non-parametric model that outputs a Gaussian predictive distribution over every data point, providing an uncertainty estimate for each prediction \citep{Obrezanova2007}. GPs have been known to match or exceed the predictive performance of traditional QSAR models such as support vector machines (SVMs) or partial least squares (PLS), while also providing insight into the reliability of those predictions \citep{Obrezanova2007}. The GP defines a distribution over functions, whose structure is determined by the kernel. These kernels can be modified to fit specific domains or functionalities. Typically, a Tanimoto kernel is used when working with molecular fingerprints \citep{Ralaivola2005, moss2020, gauche}. One of the major limitations to using GPs in practice is computational scaling. Inference has $O(N^3)$ time complexity and $O(N^2)$ memory, where $N$ is the number of training points \citep{Rasmussen2005}. This can be mitigated with sparse GP approximations, but these introduce approximation error and can be more difficult to tune \citep{quinonero2005}. Ensembles of trees or BNNs are typically more efficient for larger data sizes \citep{lakshminarayanan2017}.

We also employed GP models using the \texttt{Gauche} framework \citep{gauche}, which is optimized for cheminformatics tasks. Standardized kernels for GPs are not optimized for the chemical landscape. \texttt{Gauche} defines such kernels for strings, fingerprints, and graphs that operate on a range of widely used molecular representations \citep{gauche}. In this study, we used the Tanimoto kernel, which is defined for binary vectors $\mathbf{x}, \mathbf{x}' \in \{0, 1\}^d$ for $d \geq 1$ as:

$$
k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x}') := \sigma_f^2 \cdot \frac{\langle \mathbf{x}, \mathbf{x}' \rangle}{\|\mathbf{x}\|^2 + \|\mathbf{x}'\|^2 - \langle \mathbf{x}, \mathbf{x}' \rangle},
$$

where $\|\cdot\|$ represents the Euclidean norm \citep{gauche}. 

% TODO [REPETITION]: The NN/BNN paragraph below is nearly identical to the one in the Introduction.
% RECOMMENDATION: Keep the detailed version here; trim the intro to a brief motivation.
NNs are a popular choice in QSAR research, though they are often outperformed by more traditional methods \citep{Baskin2008, Koutsoukas2017}. Decision trees and related ensemble models tend to perform stronger on smaller datasets, overfit less, and don't rely on a static 3D molecular structure like NNs do \citep{Baskin2008, Koutsoukas2017}. NNs are often deterministic; however, they can be transformed into probabilistic models through several approaches. BNNs introduce priors on the weights and compute a posterior distribution on those weights given the input data, providing uncertainty quantification. This can be done by approximations, including Monte Carlo dropout, variational inference, or ensembles \citep{gal2016}. To improve computational efficiency, several strategies exist for converting NNs to BNNs. One straightforward yet computationally expensive approach is to replace all linear layers with Bayesian layers (full Bayesian transformation, or full-BNN), using Gaussian priors $\mathcal{N}(0, 0.1^2)$ on all weights. Replacing only the final layer is computationally lighter (last-layer transformation, or last-layer-BNN), using the same priors on the final layer only. A third approach is Variational Bayesian Last Layers (VBLL), which maintains a mean-field variational posterior $q(\mathbf{W}) = \mathcal{N}(\boldsymbol{\mu}_W, \text{diag}(\boldsymbol{\sigma}_W^2))$ over the last-layer weights and is trained by maximizing the evidence lower bound (ELBO), i.e., minimizing the reconstruction loss plus KL divergence $D_{\text{KL}}(q(\mathbf{W}) \| p(\mathbf{W}))$ from a standard normal prior, scaled by $1/N$ \citep{Harrison2024} (variational transformation, or var-BNN). The VBLL also learns a scalar observation noise variance, enabling decomposition of predictive uncertainty into epistemic (from weight posterior sampling) and aleatoric (from learned noise) components. All BNN variants use 100 Monte Carlo forward passes at inference to estimate predictive distributions. Bayesian approaches also act as forms of regularization \citep{Burden2009}, as dropouts help to prevent overfitting.

% TODO [MISSING FILE]: bayesian_transformation.png does not exist yet. Create methods diagram
% showing deterministic vs BNN architecture (fixed weights vs weight distributions).
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{bayesian_transformation.png}
    \caption{Deterministic versus Bayesian neural network architectures. Deterministic networks use fixed weights $W$ to produce point predictions, while Bayesian networks treat weights as distributions $\mathcal{N}(\mu, \sigma^2)$, generating prediction histograms through multiple stochastic forward passes.}
    \label{fig:bayesian_transformation}
\end{figure}

% TODO [REPETITION]: The Bayesian methods paragraph below repeats material from the Introduction.
% RECOMMENDATION: Remove this paragraph entirely; it adds nothing beyond what the intro already says.
Unlike standard deterministic ML methods, Bayesian methods model distributions over predictions using Bayes' rule, allowing for the quantification of uncertainty within parameters and labels. Bayesian methods treat the model parameters as random variables with distributions. Uncertainty can also be estimated from other non-deterministic methods, including ensemble \citep{lakshminarayanan2017}, bootstrap modeling \citep{palmer2022}, and federated learning \citep{Hanser2023}.

We used the following deterministic NNs: standard feed-forward NNs and multi-layer perceptron (MLP) networks. To transform these into probabilistic NN variants, we implemented BNNs using three approaches: full Bayesian transformation (full-BNN) with weight uncertainty propagated through all layers, last-layer transformation (last-layer-BNN) where only the final layer is treated probabilistically, and variational transformation (var-BNN) approximations \citep{Harrison2024}. Full-BNN implementation is visualized in Figure~\ref{fig:bayesian_transformation}. We implemented NN and MLP models using \texttt{PyTorch} \citep{pytorchGeometric} with ReLU activations and dropout regularization ($p=0.2$). The DNN architecture uses two hidden layers of sizes [128, 64]. Each model was trained using the Adam optimizer with early stopping on validation loss.

% TODO: Confirm conformal prediction details and citations from previous paper versions
We also applied conformal prediction as a model-agnostic approach to uncertainty quantification \citep{vovk2005}. Split conformal prediction uses a held-out calibration set to compute nonconformity scores from the base model's residuals, then constructs prediction intervals by taking the appropriate quantile of these scores \citep{lei2018}. This provides distribution-free coverage guarantees without assumptions about the underlying data distribution. We applied conformal prediction to RF, QRF, XGBoost, DNN, and Gauche GP base models, using calibration sets of varying sizes to assess the effect of calibration data on interval quality.

\subsection{Performance Metrics}

% TODO: explain the possible values of R2 and what they mean 
This paper evaluates performance and noise robustness across a range of metrics. Root mean squared error (RMSE) and the coefficient of determination ($R^2$) are primarily used to compare QSAR performance. $\sigma$ denotes the relative noise level for each experiment, standardized across noise strategies. 

Pairwise statistical comparisons between deterministic and probabilistic architecture counterparts (e.g., DNN vs DNN-BNN variants, RF vs QRF) were conducted using the Wilcoxon signed-rank test, with $\alpha = 0.05$ (two-sided). To assess whether model rankings remain consistent across different noise injection strategies, we computed Kendall's coefficient of concordance (Kendall's W), which ranges from 0 (no agreement) to 1 (complete agreement). Values above 0.7 indicate strong agreement in rankings across strategies.

When evaluating uncertainty, we assess the relationship between predicted uncertainty and prediction quality. Spearman's rank correlation coefficient ($\rho$) was used to quantify the relationship between predicted uncertainty ($u_i$) and absolute error ($|y_i - \hat{y}_i|$), as well as between predicted uncertainty and injected noise magnitude. Higher correlations indicate that the model's uncertainty estimates reliably identify predictions that are likely to be incorrect.

For probabilistic models, \textbf{empirical coverage} was evaluated at $1\sigma$ and $2\sigma$ confidence intervals by computing the proportion of test predictions where the true value falls within $\hat{y}_i \pm k\hat{u}i$:
$$
\text{Coverage}(k\sigma) = \frac{1}{N}\sum{i=1}^{N} 1\big[|y_i - \hat{y}_i| \leq k\hat{u}_i\big],
$$
such that $\hat{u}_i$ is the predicted uncertainty for sample $i$ and $k \in {1, 2}$. Theoretical target coverages are defined as 68\% at $1\sigma$ and 95\% at $2\sigma$ for Gaussian distributions.

The \textbf{expected calibration error} (ECE) was computed by binning predictions into deciles by predicted uncertainty and measuring the weighted average absolute difference between predicted uncertainty and observed error:
$$
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \big| \bar{u}_b - \bar{e}_b \big|,
$$
where $B_b$ is the set of predictions in bin $b$, $\bar{u}_b$ is the mean predicted uncertainty in that bin, and $\bar{e}_b$ is the mean absolute error. Lower ECE indicates better-calibrated uncertainty estimates.


To evaluate the effect of noise, we examined model performance degradation under increasing noise with $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$ using the Noise Degradation Slope (NDS), computed as the slope of R$^2$ versus noise level:
$$
\text{NDS} = \frac{dR^2}{d\sigma}.
$$

% TODO: Results section and Table 3 use threshold 0.6, but analysis code uses 0.5 (BASELINE_THRESHOLD in generate_paper_figures.py). Reconcile these after seeing results. Current text matches Results section.
Values closer to zero indicate that noise has minimal effect on performance, negative values indicate a higher sensitivity to noise. Positive values, which would not observed in this study, would indicate that the noise improves model performance. Configurations with baseline R$^2 \leq 0.6$ were excluded from robustness analysis, as poor performance on clean labels tends to remain poor for noisy labels as well, producing misleadingly shallow slopes.

A two-way analysis of variance (ANOVA) variance decomposition was conducted separately for each noise strategy to identify the relative contributions of molecular representation and model architecture choice on both prediction performance and noise robustness. This per-strategy approach avoids inappropriate aggregation across fundamentally different noise types. For a given metric $y$ (either $R^2$ at fixed noise or NDS):
$$
y_{ijr} = \mu + \alpha_i + \beta_j + (\alpha\beta){ij} + \epsilon{ijr},
$$
such that $\alpha_i$ represents the effect of model architecture $i$, $\beta_j$ represents the effect of molecular representation $j$, $(\alpha\beta){ij}$ is the interaction term, and $\epsilon{ijr}$ is the residual for replicate $r$.

The proportion of variance explained by each factor was calculated as the effect size $\eta^2$:
$$
\eta^2_{\text{factor}} = \frac{SS_{\text{factor}}}{SS_{\text{total}}},
$$
using Type I (sequential) sum of squares, such that $SS$ denotes the sum of squares. Both factors (model architecture and molecular representation) are treated as fixed effects, and the 10 experimental replicates per cell provide the error term.

To ensure a valid fully crossed design, we curated the set of ANOVA factor levels by removing redundant or near-duplicate levels. Pairwise Spearman rank correlations were computed between all model NDS profiles and between all representation NDS profiles across noise strategies. Models whose NDS profiles correlated at $\rho > 0.99$ with another model were excluded: this removed conformal prediction wrappers (conformal RF, conformal QRF, conformal DNN), which produced near-identical robustness profiles to their base models, and quantile regression forests (QRF), which were highly redundant with RF. Among representations, SNS and ECFP4 correlated at $\rho > 0.90$, and SNS was excluded to avoid inflating representation degrees of freedom. We also computed intraclass correlation coefficients ICC(1,1) for all model pairs to assess within-family consistency; ICC(1,1) measures the proportion of total variance attributable to between-subject (i.e., between-configuration) differences, with values near 1.0 indicating that two models rank configurations almost identically. The full redundancy and ICC tables are reported in Supplementary Tables S5--S7. After exclusion, the ANOVA retained all remaining model architectures and five representations (ECFP4, PDV, SMILES, mol2vec, and MHGGNN), yielding a fully crossed design.

To ensure fair model comparisons, main analyses use a single primary representation (physicochemical descriptor vectors, PDV) rather than aggregating across representations with different baseline performance levels.

\subsection{Noise Strategies}
In this research, our objective is to evaluate the robustness of QSAR models against label noise. To do so, we inject artificial noise into the labels during training. Validation and test data remain free of noise. This artificial noise simulates real-world experimental noise. The clean target label $y_i$ is replaced by a noisy label $\tilde{y}_i = y_i + \epsilon_i$, where $\epsilon_i$ is the artificial noise injected at the index $i$, the value of which is determined by a specified noise strategy.  

Experimental noise is often modeled as homoscedastic Gaussian noise and is added evenly across all labels. However, not all experimental noise is random. It may be heteroscedastic, where the variance depends on the molecule itself or experimental factors, or systematic, where biases from factors like assay conditions are introduced. We used a variety of noise strategies to model the different types of possible experimental noise that could arise in cheminformatics data. 

The following strategies were implemented for regression (Table~\ref{tab:regression_noise}, Figure~\ref{fig:noise_strategies}):
\begin{table}[h]
\centering
\begin{tabular}{lll}
\toprule
\textbf{Noise Type} & \textbf{Noise Scaling} & \textbf{Simulated Real-World Source} \\
\midrule
Legacy & $\sigma$ (fixed) & Random measurement error \\
Outlier & $3\sigma$ / $0.1\sigma$ & Transcription errors, batch effects \\
Quantile & $2\sigma$ / $0.1\sigma$ & Systematic error in hard-to-predict domains \\
Hetero & $\sigma\sqrt{0.1 + 0.05|y|}$ & Heteroscedastic measurement precision \\
Threshold & $2\sigma$ / $0.1\sigma$ & Regime-dependent assay errors \\
Valprop & $\sigma(1 + 0.05|y|)$ & Percentage-based measurement uncertainty \\
\bottomrule
\end{tabular}
\caption{Regression noise injection strategies and their real-world analogues.}
\label{tab:regression_noise}
\end{table}

The parameter $\sigma$ serves as a noise scaling factor across all strategies. For legacy noise, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ is applied uniformly. For outlier noise, samples identified as outliers ($z$-score $> 2.0$) receive noise from $\mathcal{N}(0, (3\sigma)^2)$ while normal samples receive $\mathcal{N}(0, (0.1\sigma)^2)$. For quantile noise, samples above the 90th or below the 10th percentile of the target distribution receive noise scaled by $2.0\times$, while central samples receive $0.1\times$. For threshold noise, samples with $|y| > 1.0$ (on normalized data) receive the higher $2.0\times$ multiplier, while those within receive $0.1\times$. For value-proportional, $\epsilon_i \sim \mathcal{N}(0, (\sigma(1 + 0.05|y_i|))^2)$, using a multiplicative formulation where noise scales proportionally with the absolute target value. And finally, for heteroscedastic, noise variance is computed as $\alpha \sigma^2 + \beta \sigma^2 |y_i|$ with $\alpha = 0.1$ and $\beta = 0.05$. All other multipliers remained constant, meaning that while the relative noise distribution differs between strategies, the difficulty scaling controlled by $\sigma$ is consistent.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_methods_noise_strategies.png}
\caption{Effect of each noise injection strategy on the label distribution at $\sigma = 0.5$. Gray histograms show the clean HOMO-LUMO gap distribution; colored histograms show the distribution after noise injection. Each strategy produces a distinct perturbation pattern, ranging from uniform displacement (Gaussian) to targeted corruption of specific label regions (quantile, threshold, outlier). The detailed progression across noise levels is shown in Supplementary Figure~\ref{fig:supp_noise_detailed}.}
\label{fig:noise_strategies}
\end{figure}

\subsubsection{NoiseInject Framework}

In an extension of this work, we developed an open-source benchmarking framework called NoiseInject to evaluate model robustness and uncertainty quantification in the presence of artificial noise for both regression and classification tasks. The framework accepts datasets in standard formats (CSV, DataFrame, or NumPy arrays) and applies artificial noise to labels while preserving the integrity of the test set.

The framework computes standard regression metrics (RMSE, $R^2$, mean absolute error [MAE]) and classification metrics (accuracy, F1 score, ROC-AUC) along with uncertainty-specific measures: expected calibration error, coverage of the prediction interval (or prediction set for classification) at specified confidence levels, mean interval width, and Pearson correlation between predicted uncertainty and absolute error. We included noise robustness metrics including noise degradation slope and retention percentage to quantify performance degradation rates. All experiments are logged to JSON Lines format with full metadata (noise configuration, model identifiers, hyperparameters, random seeds).

The evaluation pipeline is model-agnostic, although reference implementations of GPs (using \texttt{Gauche} \citep{gauche}), split-CP for sklearn-compatible models, and Monte Carlo dropout wrappers for PyTorch networks are provided. We added visualization tools which generate performance degradation curves, calibration plots, and uncertainty-error scatter plots for rapid assessment of model behavior under noise. 

The software is available under an MIT license at \url{github.com/adpunt/noise\_inject} and is designed to integrate into existing cheminformatics or other data pipelines, enabling systematic benchmarking of model robustness.


\section{Results}

\subsection{The Role Inversion: Performance vs Robustness}\label{sec:role_inversion}

A central finding of this study is that the factors governing predictive performance differ fundamentally from those governing noise robustness. ANOVA decomposition across six noise strategies reveals a consistent pattern (Table~\ref{tab:anova_decomposition}, Figure~\ref{fig:anova_decomposition}): for predictive performance (R$^2$ at $\sigma = 0.3$), model architecture and molecular representation contribute roughly equally ($\eta^2 \approx 20$--$29\%$ each), with interaction and residual terms accounting for the remainder. For noise robustness (NDS), model architecture becomes the dominant factor ($\eta^2 = 25.6$--$47.6\%$), while representation recedes to a minor role ($\eta^2 = 2.0$--$14.9\%$).

This role inversion has a natural interpretation. For performance, molecular representation determines what chemical information is available to the model, while architecture determines how efficiently it is used---both matter roughly equally. For robustness, representation becomes less important because noise corrupts labels, not features. Instead, the model's inductive biases---how it regularizes, ensembles, or distributes uncertainty---determine whether corrupted labels degrade predictions or are absorbed by the learning algorithm.

The exception is outlier noise, where model architecture explained only $12.1\%$ of robustness variance and neither factor was dominant. This is consistent with outlier noise being the mildest strategy (mean $|$NDS$| = 0.099$): when degradation is negligible, there is little variance for any factor to explain. The strongest model effects emerged under threshold ($\eta^2 = 46.5\%$) and value-proportional ($\eta^2 = 47.6\%$) noise---the two most severe strategies---indicating that structured noise is most revealing of architectural differences.

\begin{table}[htbp]
\centering
\caption{\textbf{ANOVA variance decomposition by noise strategy.} $\eta^2$ values (\%) for model architecture and molecular representation effects on predictive performance (R$^2$ at $\sigma=0.3$) and noise robustness (NDS). Interaction and residual terms account for the remaining variance.}
\label{tab:anova_decomposition}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Performance ($\eta^2$, \%)}} & \multicolumn{2}{c}{\textbf{Robustness ($\eta^2$, \%)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Strategy} & \textbf{Model} & \textbf{Rep} & \textbf{Model} & \textbf{Rep} \\
\midrule
Gaussian        & 20.6 & 27.5 & 40.8 &  9.1 \\
Quantile        & 22.0 & 28.3 & 33.3 &  3.5 \\
Threshold       & 20.9 & 29.6 & 46.5 & 14.9 \\
Heteroscedastic & 25.0 & 25.6 & 25.6 &  3.4 \\
Value-prop.     & 28.0 & 20.9 & 47.6 & 11.9 \\
Outlier         & 20.6 & 29.6 & 12.1 &  2.0 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig2_anova_decomposition.png}
    \caption{\textbf{ANOVA variance decomposition by noise strategy.} Variance explained ($\eta^2$, \%) for (A) predictive performance (R$^2$ at $\sigma=0.3$) and (B) noise robustness (NDS), decomposed into model architecture, molecular representation, and interaction effects for each noise strategy.}
    \label{fig:anova_decomposition}
\end{figure}

To ensure meaningful robustness comparisons, we restricted analysis to configurations achieving baseline R$^2 > 0.6$. Configurations with poor baseline performance can exhibit misleadingly shallow degradation slopes simply because there is less performance to lose. This threshold excluded 63 configurations, of which 16 were marginal ($0.5 \leq R^2 < 0.6$). The majority of exclusions were BNN variants on mol2vec and MHG-GNN representations, where the Bayesian transformation produced near-zero baseline R$^2$ ($\approx -0.02$); these appear as ``N/A'' in supplementary tables and heatmaps. Additionally, 32 individual training iterations were excluded due to catastrophic failures (R$^2 < -0.5$), primarily affecting DNN on mol2vec (Supplementary Table~\ref{tab:excluded_configs}).

\subsection{Model Rankings and Noise Strategy Patterns}\label{sec:rankings}

We evaluated all model architectures under increasing noise on the PDV representation across six strategies (Figure~\ref{fig:global_overview}). Performance degrades approximately linearly with noise level for all models, but the rate of degradation---quantified by the Noise Degradation Slope (NDS)---varies substantially across architectures.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig1_global_overview.png}
    \caption{\textbf{Global overview of noise robustness.} (A) Performance degradation curves (PDV, Gaussian noise) showing R$^2$ versus noise level $\sigma$ for representative models. (B) NDS heatmap across all model architectures and noise strategies on PDV; darker cells indicate steeper degradation.}
    \label{fig:global_overview}
\end{figure}

Averaging across all six strategies, NGBoost achieved the best mean robustness (mean NDS $= -0.334$), followed by SVM ($-0.358$) and DNN-BNN-Full ($-0.367$). QRF ($-0.407$) and MLP ($-0.406$) were the least robust overall (Table~\ref{tab:nds_ranking}). Model rankings across noise strategies were highly consistent: Kendall's coefficient of concordance yielded $W = 0.957$ ($\chi^2 = 80.40$, $p = 2.4 \times 10^{-11}$; 15 ANOVA-included models, 6 strategies), indicating that a model's relative robustness is largely strategy-independent (Supplementary Figure~\ref{fig:supp_ranking_consistency}). The main exception is outlier noise, where uniformly mild degradation compresses rankings and makes models harder to distinguish. Parallel analysis on the ECFP4 representation confirmed the same patterns (Supplementary Figures~\ref{fig:supp_ecfp4_overview} and~\ref{fig:supp_ecfp4_ranking}).

\begin{table}[htbp]
\centering
\caption{\textbf{Noise Degradation Slope by model on PDV.} Representative models ranked by mean NDS across all six strategies. Lower $|$NDS$|$ indicates greater noise robustness. Conformal wrappers, QRF, and flexible DNN architectural variants are excluded for clarity; full results in Supplementary Table~\ref{tab:nds_all_reps}.}
\label{tab:nds_ranking}
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Gauss.} & \textbf{Outlier} & \textbf{Quant.} & \textbf{Thresh.} & \textbf{Hetero.} & \textbf{Val.-P.} & \textbf{Mean} \\
\midrule
NGBoost             & $-0.31$ & $-0.10$ & $-0.27$ & $-0.61$ & $-0.17$ & $-0.54$ & $-0.33$ \\
SVM                 & $-0.34$ & $-0.10$ & $-0.28$ & $-0.67$ & $-0.18$ & $-0.58$ & $-0.36$ \\
DNN-BNN (Full)      & $-0.34$ & $-0.13$ & $-0.30$ & $-0.66$ & $-0.19$ & $-0.59$ & $-0.37$ \\
MLP-BNN (Full)      & $-0.35$ & $-0.12$ & $-0.30$ & $-0.66$ & $-0.19$ & $-0.59$ & $-0.37$ \\
LightGBM            & $-0.35$ & $-0.11$ & $-0.30$ & $-0.69$ & $-0.19$ & $-0.61$ & $-0.38$ \\
XGBoost             & $-0.35$ & $-0.12$ & $-0.31$ & $-0.69$ & $-0.19$ & $-0.62$ & $-0.38$ \\
GP (Gauche)         & $-0.36$ & $-0.12$ & $-0.32$ & $-0.69$ & $-0.20$ & $-0.62$ & $-0.39$ \\
DNN-BNN (Last)      & $-0.36$ & $-0.13$ & $-0.33$ & $-0.69$ & $-0.22$ & $-0.62$ & $-0.39$ \\
DNN                 & $-0.36$ & $-0.13$ & $-0.33$ & $-0.70$ & $-0.20$ & $-0.63$ & $-0.39$ \\
RF                  & $-0.36$ & $-0.12$ & $-0.31$ & $-0.73$ & $-0.20$ & $-0.64$ & $-0.39$ \\
MLP-BNN (Last)      & $-0.39$ & $-0.14$ & $-0.34$ & $-0.70$ & $-0.22$ & $-0.64$ & $-0.40$ \\
MLP                 & $-0.37$ & $-0.14$ & $-0.35$ & $-0.72$ & $-0.21$ & $-0.65$ & $-0.41$ \\
\bottomrule
\end{tabular}
\end{table}

The six noise strategies produced markedly different absolute degradation but fell into three severity tiers (Table~\ref{tab:phase5_noise_strategy}). \textbf{Mild}: outlier noise (NDS range: $-0.10$ to $-0.14$) and heteroscedastic noise ($-0.17$ to $-0.22$), where models retained $>93\%$ of baseline performance. \textbf{Moderate}: quantile ($-0.27$ to $-0.35$) and Gaussian ($-0.31$ to $-0.39$). \textbf{Severe}: value-proportional ($-0.54$ to $-0.65$) and threshold ($-0.61$ to $-0.73$), where models lost $>34\%$ of baseline performance. Outlier noise was benign because ensemble methods can isolate aberrant observations; threshold noise was the most destructive because it deterministically corrupts all labels above a property cutoff, introducing systematic bias that models cannot learn around.

\begin{table}[htbp]
\centering
\caption{\textbf{Noise strategy severity.} Mean performance retention and NDS across all ANOVA-included configurations.}
\label{tab:phase5_noise_strategy}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Noise} & \textbf{Mean} & \textbf{Mean} & \textbf{Std} & \textbf{Mean} \\
\textbf{Strategy} & \textbf{Baseline $R^2$} & \textbf{Retention \%} & \textbf{Retention \%} & \textbf{$|$NDS$|$} \\
\midrule
Outlier & 0.778 & 97.5 & 0.3 & 0.099 \\
Heteroscedastic & 0.776 & 93.4 & 0.7 & 0.194 \\
Quantile & 0.785 & 88.6 & 1.5 & 0.286 \\
Gaussian & 0.831 & 81.0 & 3.3 & 0.362 \\
Value-proportional & 0.785 & 65.9 & 3.6 & 0.612 \\
Threshold & 0.785 & 57.8 & 4.7 & 0.675 \\
\bottomrule
\end{tabular}
\end{table}

Despite the variation in absolute NDS values, the \emph{relative} severity of noise strategies was remarkably uniform across architectures. Strategy sensitivity ratios---each model's NDS on a given strategy divided by its Gaussian NDS---varied by less than $10\%$ across all models for every strategy (Table~\ref{tab:strategy_sensitivity}). Threshold noise was approximately $1.9$--$2.0\times$ as damaging as Gaussian for every model; value-proportional $1.60$--$1.77\times$; outlier $0.29$--$0.37\times$. This uniformity indicates that the relative severity of noise strategies is a property of the noise structure itself, not of model-specific vulnerabilities. Models differ in how much they degrade overall, but the proportional scaling across noise types is constant---a finding that simplifies noise robustness benchmarking.

\begin{table}[htbp]
\centering
\caption{\textbf{Strategy sensitivity ratios relative to Gaussian noise.} Each value represents a model's NDS on the given strategy divided by its Gaussian NDS; values $> 1$ indicate greater damage than Gaussian, $< 1$ indicate less. Ratios are strikingly uniform across architectures, indicating that relative strategy severity is noise-intrinsic.}
\label{tab:strategy_sensitivity}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Outlier} & \textbf{Quantile} & \textbf{Threshold} & \textbf{Hetero.} & \textbf{Val.-Prop.} \\
\midrule
NGBoost  & 0.33 & 0.87 & 1.97 & 0.55 & 1.75 \\
SVM      & 0.30 & 0.84 & 1.98 & 0.55 & 1.72 \\
LightGBM & 0.32 & 0.86 & 1.98 & 0.55 & 1.75 \\
XGBoost  & 0.35 & 0.88 & 1.98 & 0.54 & 1.77 \\
RF       & 0.32 & 0.86 & 2.00 & 0.54 & 1.76 \\
GP       & 0.34 & 0.87 & 1.91 & 0.56 & 1.71 \\
DNN      & 0.37 & 0.91 & 1.93 & 0.55 & 1.74 \\
MLP      & 0.37 & 0.92 & 1.92 & 0.55 & 1.74 \\
\bottomrule
\end{tabular}
\end{table}

Baseline performance (R$^2$ at $\sigma = 0$) does not strongly predict noise robustness: models with similar clean-data performance can exhibit markedly different degradation slopes (Figure~\ref{fig:full_overview}), underscoring that robustness is a distinct property from accuracy. Sigma-level rankings (Supplementary Table~\ref{tab:sigma_rankings}) reveal this trade-off most clearly: NGBoost ranks 19th in clean performance but 5th at $\sigma = 1.0$, while QRF ranks 5th clean but drops to 17th under heavy noise. SVM shows a similar trajectory, rising from 11th clean to 4th at $\sigma = 1.0$. Models with inherent noise robustness tend to sacrifice clean-data accuracy---a trade-off that only becomes apparent under systematic noise evaluation.

\subsection{Three Mechanisms of Noise Robustness}\label{sec:mechanisms}

The overall ANOVA establishes that model architecture is the dominant factor for robustness, but simple effects analysis (Supplementary Table~\ref{tab:simple_effects}) reveals that different models achieve robustness through qualitatively different mechanisms, depending on how much their noise robustness depends on the choice of representation.

\paragraph{Inherent robustness.} SVM and the full Bayesian neural network variants show robustness that is largely representation-independent. Within SVM, representation explains only $11.8\%$ of robustness variance under Gaussian noise ($p = 0.23$), $12.4\%$ under threshold ($p = 0.21$), and $13.4\%$ under value-proportional ($p = 0.16$)---none significant. Full BNNs show a similar pattern: DNN-BNN-Full's representation dependence is $15.3\%$ under Gaussian ($p = 0.11$) and $13.3\%$ under threshold ($p = 0.14$). These models achieve noise robustness through their inductive biases---SVM's margin maximization and the BNN's weight priors---rather than by relying on a favorable input representation.

\paragraph{Representation-mediated robustness.} RF, MLP, and the last-layer BNN variants show the opposite pattern: their robustness depends heavily on which representation they receive. Within MLP, representation explains $60.2\%$ of robustness variance under Gaussian noise ($p = 1.4 \times 10^{-8}$) and $85.5\%$ under threshold ($p = 2.9 \times 10^{-18}$). RF shows $69.4\%$ under Gaussian ($p = 2.5 \times 10^{-10}$) and $83.4\%$ under threshold ($p = 7.7 \times 10^{-16}$). These models can be robust, but only when paired with the right representation---on the wrong representation, they degrade steeply.

\paragraph{The BNN natural experiment.} Full versus last-layer Bayesian transformation provides a controlled comparison: same base architecture, same training data, same representations, differing only in the extent of Bayesian treatment. The contrast is stark. For DNN-BNN-Last, representation explains $57.7\%$ of robustness variance under Gaussian noise; for DNN-BNN-Full, only $15.3\%$. For MLP variants, the difference is even larger: $79.6\%$ (Last) versus $28.0\%$ (Full). Full Bayesian treatment transforms a representation-mediated architecture into an inherently robust one by placing priors on all weights, not just the output layer.

This mechanistic difference is visible in the degradation curves (Figures~\ref{fig:dnn_family} and~\ref{fig:mlp_rf_comparison}) and reflected in paired statistical comparisons (Table~\ref{tab:wilcoxon_bnn}). DNN-BNN-Full was significantly more robust than deterministic DNN ($p = 1.2 \times 10^{-7}$, mean NDS improvement $+0.067$), while DNN-BNN-Last showed no significant improvement ($p = 0.105$). The same pattern held for MLP: MLP-BNN-Full improved significantly ($p = 1.2 \times 10^{-7}$, improvement $+0.115$), MLP-BNN-Last did not ($p = 0.248$). The benefit of full Bayesian transformation also varies by representation (Table~\ref{tab:nn_transforms}): on SMILES, where the base MLP struggles most, MLP-BNN-Full improves NDS by $+0.217$; on ECFP4, where the base model is already adequate, the improvement is only $+0.094$.

\begin{table}[htbp]
\centering
\caption{\textbf{Wilcoxon signed-rank tests for Bayesian and probabilistic transformations.} Paired comparisons of NDS across matched model--representation--strategy configurations. Positive improvement indicates the variant is more robust than its deterministic counterpart.}
\label{tab:wilcoxon_bnn}
\small
\begin{tabular}{llrrrl}
\toprule
\textbf{Family} & \textbf{Comparison} & \textbf{$n$} & \textbf{$\Delta$ NDS} & \textbf{$p$-value} & \textbf{Sig.} \\
\midrule
DNN & DNN vs DNN-BNN (Full) & 24 & $+0.067$ & $1.2 \times 10^{-7}$ & * \\
DNN & DNN vs DNN-BNN (Last) & 22 & $+0.004$ & $0.105$ & \\
MLP & MLP vs MLP-BNN (Full) & 24 & $+0.115$ & $1.2 \times 10^{-7}$ & * \\
MLP & MLP vs MLP-BNN (Last) & 22 & $+0.003$ & $0.222$ & \\
RF  & RF vs QRF             & 20 & $-0.021$ & $2.7 \times 10^{-5}$ & * \\
\bottomrule
\end{tabular}
\end{table}

For random forests, QRF was significantly \emph{less} robust than RF (Wilcoxon $p = 2.7 \times 10^{-5}$, mean NDS worsening $-0.021$). The QRF's quantile regression mechanism does not confer noise robustness; instead, the additional flexibility may cause overfitting to noisy labels.

% TODO [DATA PENDING]: fig4_dnn_family.png and fig5_mlp_rf_comparison.png may update once BNN variational data completes.
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig4_dnn_family.png}
    \caption{\textbf{DNN family comparison.} R$^2$ versus noise level $\sigma$ for the DNN and its BNN variants (full and last-layer) under Gaussian and Heteroscedastic noise (PDV representation).}
    \label{fig:dnn_family}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig5_mlp_rf_comparison.png}
    \caption{\textbf{MLP and RF family comparison.} R$^2$ versus noise level $\sigma$ for (A--B) MLP and its BNN variants and (C--D) RF versus QRF, under Gaussian and Heteroscedastic noise (PDV representation).}
    \label{fig:mlp_rf_comparison}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Neural network Bayesian transformation effects by representation.}
\label{tab:nn_transforms}
\small
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Representation} & \textbf{Baseline R$^2$} & \textbf{MLP NDS} & \textbf{BNN-Full NDS} & \textbf{NDS Improvement} \\
\midrule
MLP & ECFP4 & 0.790 & $-0.495$ & $-0.402$ & +0.094 \\
MLP & PDV & 0.828 & $-0.406$ & $-0.369$ & +0.037 \\
MLP & SNS & 0.813 & $-0.546$ & $-0.434$ & +0.112 \\
MLP & SMILES & 0.728 & $-0.600$ & $-0.383$ & +0.217 \\
\bottomrule
\end{tabular}
\end{table}

The representation perspective complements the model perspective. Some representations amplify model differences: within SMILES, model architecture explains $91.3\%$ of robustness variance under Gaussian noise---choosing the wrong model with SMILES input incurs a severe robustness penalty. Within ECFP4, model explains $74.6\%$. Other representations compress these differences: within PDV, model explains only $40.1\%$, and within mol2vec $41.3\%$. PDV acts as an equalizer, producing moderate robustness regardless of model choice, while SMILES acts as an amplifier, making model selection critical. These interaction patterns are visible in Figure~\ref{fig:interaction}A, where model rankings change substantially across representations, and Figure~\ref{fig:full_overview}, where the vertical spread of points for each model shows its representation sensitivity. The complete NDS values for all model--representation--strategy combinations are provided in Supplementary Table~\ref{tab:nds_all_reps}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig_full_overview.png}
    \caption{\textbf{All model--representation configurations across noise severity tiers.} Each point is one model--representation configuration; marker shape indicates representation, color indicates model. Panels show (A) Gaussian, (B) outlier, and (C) threshold noise strategies, spanning the mild-to-severe range. Configurations from the remaining strategies are shown in Supplementary Figure~\ref{fig:full_overview_supp}.}
    \label{fig:full_overview}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig_interaction.png}
    \caption{\textbf{Representation--model interaction effects.} (A) NDS heatmap showing model robustness across representations under Gaussian noise. (B) NDS on PDV versus ECFP4 for each model under Gaussian noise, with Spearman correlation.}
    \label{fig:interaction}
\end{figure}

\subsection{Uncertainty as the Mechanistic Link}\label{sec:uncertainty}

The three robustness mechanisms suggest a deeper question: \emph{why} do some architectures resist noise? We hypothesized that models whose uncertainty estimates respond to injected noise would also maintain prediction quality under noise---that noise detection and noise resistance are linked. Among the seven uncertainty-producing models, this link is strong: noise detection ability (Unc-Noise $\rho$ in Table~\ref{tab:uncertainty_metrics}) closely mirrors noise robustness (mean NDS), with a Spearman correlation of $\rho \approx 0.90$ ($p < 0.01$, $n = 7$). Models that detect noise also resist it.

% TODO [DATA PENDING]: Update table once BNN variational uncertainty data is available.
% Models with mean uncertainty < 1e-3 (deterministic DNN/MLP) are excluded.
\begin{table}[htbp]
\centering
\caption{Uncertainty quantification metrics for probabilistic models under Gaussian noise (ECFP4 representation). Unc-Error $\rho$: Spearman correlation between predicted uncertainty and absolute prediction error. Unc-Noise $\rho$: Spearman correlation between predicted uncertainty and injected noise magnitude. ECE: Expected calibration error. Coverage: proportion of true values within the predicted interval at $1\sigma$ (target: 68\%) and $2\sigma$ (target: 95\%).}
\label{tab:uncertainty_metrics}
\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Model} & \textbf{Unc-Error $\rho$} & \textbf{Unc-Noise $\rho$} & \textbf{ECE} & \textbf{Cov. $1\sigma$} & \textbf{Cov. $2\sigma$} \\
\midrule
NGBoost        & 0.22 & 0.38 & 0.12 & 69.4\% & 95.1\% \\
QRF            & 0.22 & 0.04 & 0.16 & 69.0\% & 91.7\% \\
GP (Gauche)    & 0.20 & 0.33 & 0.20 & 74.6\% & 94.8\% \\
MLP-BNN (Full) & 0.19 & 0.32 & 0.23 & 74.9\% & 94.9\% \\
DNN-BNN (Full) & 0.17 & 0.34 & 0.19 & 73.9\% & 94.7\% \\
MLP-BNN (Last) & 0.11 & 0.23 & 0.14 & 56.5\% & 82.4\% \\
DNN-BNN (Last) & 0.09 & 0.15 & 0.14 & 52.4\% & 79.3\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_uncertainty_combined.png}
\caption{\textbf{Uncertainty quantification under increasing label noise} (PDV, Gaussian strategy). Mean predicted uncertainty as a function of injected noise level $\sigma$ for each probabilistic model. Models whose uncertainty grows with $\sigma$ are detecting added noise; flat lines indicate noise-insensitive estimates.}
% TODO [DATA PENDING]: Expand to 3-panel figure once BNN variational uncertainty data is available.
\label{fig:uncertainty_combined}
\end{figure}

NGBoost exemplifies this link. It produces the strongest noise tracking ($\rho = 0.38$), meaning its uncertainty estimates respond most to injected noise, and it achieves the best robustness (mean NDS $= -0.334$). It is also the best-calibrated model: $69.4\%$ coverage at $1\sigma$ (target: $68\%$) and $95.1\%$ at $2\sigma$ (target: $95\%$). At the other extreme, QRF produces uncertainty estimates that are essentially unresponsive to noise ($\rho = 0.04$) despite strong uncertainty-error correlation ($\rho = 0.22$)---its uncertainty captures model-specific variance but not data quality. This failure to detect noise maps directly onto QRF's poor robustness (mean NDS $= -0.407$, worst among all models).

The Full versus Last BNN comparison reinforces this pattern from the uncertainty side. Full BNN variants show moderate noise tracking (DNN-BNN-Full $\rho = 0.34$, MLP-BNN-Full $\rho = 0.32$) and near-target coverage ($\sim$74\% at $1\sigma$). Last-layer variants show weaker noise tracking ($\rho = 0.15$--$0.23$) and substantially undercover ($52$--$57\%$ at $1\sigma$), producing uncertainty intervals that are too narrow. The same architectural change that converts representation-mediated robustness into inherent robustness (Section~\ref{sec:mechanisms}) also converts noise-blind uncertainty into noise-responsive uncertainty---the two properties are mechanistically linked through the extent of Bayesian treatment.

Figure~\ref{fig:uncertainty_combined} illustrates this separation: NGBoost, GP, and full BNNs increase their predicted uncertainty with $\sigma$, correctly signaling degraded data quality, while last-layer BNN variants show minimal response. The practical implication is that only models whose uncertainty is ``honest''---responsive to actual data quality---provide the kind of noise resistance that generalizes.

\subsection{Generalization Across Strategies and Datasets}\label{sec:generalization}

The uniform strategy sensitivity ratios (Section~\ref{sec:rankings}) establish that robustness findings generalize well across noise types: a model's relative robustness on one strategy predicts its robustness on others ($W = 0.957$). The noise strategy acts as a scaling factor, not a differentiator---it determines how much damage occurs, but not which models handle it best.

Generalization across datasets is a different matter. We evaluated seven representative models---SVM, GP (Gauche), RF, QRF, DNN, LightGBM, and XGBoost---on three experimentally-derived molecular property datasets: LogD (lipophilicity), Caco-2 efflux permeability, and hERG-Ki (cardiac ion channel binding affinity), using all six noise strategies at $\sigma \in \{0, 0.1, \ldots, 1.0\}$. BNN variants and NGBoost were not included in external validation due to computational constraints. Configurations with $|$NDS$| > 2$ were filtered as artifacts (``N/A'' in Figure~\ref{fig:validation_overview}).

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_overview.png}
\caption{\textbf{Validation dataset NDS heatmaps.} NDS by model and noise strategy for each external dataset (LogD, Caco-2 efflux, hERG-Ki). Black cells with white ``missing'' text indicate configurations not yet run; ``N/A'' indicates filtered extreme values ($|$NDS$| > 2$).}
\label{fig:validation_overview}
\end{figure}

Validation rankings diverge substantially from QM9 (Figure~\ref{fig:validation_model_comparison}). On external datasets, SVM achieved the best mean rank (1.67 across three datasets), followed by GP/Gauche (2.33) and RF (3.0). XGBoost, which ranked second on QM9, performed worst on validation (mean rank 6.67). The correlation between QM9 and external dataset NDS is negative and non-significant ($r = -0.30$, $p = 0.51$; Figure~\ref{fig:validation_qm9_transferability}), indicating that overall robustness rankings do not transfer.

This non-transferability is interpretable through the mechanism framework developed in Section~\ref{sec:mechanisms}. SVM, whose robustness is inherent and representation-independent, ranks first on external datasets---architectural noise resistance transfers across data domains. XGBoost, whose QM9 robustness benefits from the large dataset size (130k molecules) and well-separated scaffold structure, loses this advantage on smaller, experimentally-derived datasets with 500--2,000 samples. The mechanisms that produce robustness, not just the robustness rankings themselves, determine whether findings transfer.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{fig_validation_qm9_transferability.png}
\caption{\textbf{Robustness transferability from QM9 to external datasets.} Per-model mean NDS on QM9 versus mean NDS across three external datasets, with Pearson correlation. The negative, non-significant correlation indicates that QM9 robustness rankings do not predict external dataset robustness.}
\label{fig:validation_qm9_transferability}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_model_comparison.png}
\caption{\textbf{Model robustness across validation datasets.} Mean NDS by model for each external dataset. Bar color indicates dataset; models are ordered by mean NDS (most robust first).}
\label{fig:validation_model_comparison}
\end{figure}

Certain findings were nonetheless robust across datasets. QRF was consistently less robust than RF on every external dataset (Table~\ref{tab:validation_probabilistic}), confirming the QM9 finding. Caco-2 efflux was the most challenging dataset, with all models showing steeper degradation than on LogD or hERG-Ki (Supplementary Figures~\ref{fig:supp_validation_strategy} and~\ref{fig:supp_validation_rep}). XGBoost degraded particularly steeply on Caco-2 (NDS $\approx -1.1$ to $-1.4$), a dataset-specific weakness not predicted by its QM9 performance.

\begin{table}[htbp]
\centering
\caption{\textbf{RF vs QRF on validation datasets.} QRF is consistently less robust than RF across all external datasets.}
\label{tab:validation_probabilistic}
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Dataset} & \textbf{RF NDS} & \textbf{QRF NDS} & \textbf{$\Delta$ NDS} \\
\midrule
ChEMBL-hERG-Ki     & $-0.279$ & $-0.417$ & $-0.139$ \\
OpenADMET-Caco2    & $-0.555$ & $-0.721$ & $-0.165$ \\
OpenADMET-LogD     & $-0.111$ & $-0.202$ & $-0.091$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Conclusion}\label{sec13}
This study demonstrates that while model architecture and molecular representation contribute roughly equally to QSAR model performance, model architecture becomes the dominant factor under label noise---explaining up to 48\% of robustness variance compared to 2--15\% for representation. Simple effects analysis reveals three distinct mechanisms of noise robustness: \emph{inherent} robustness (SVM, full BNNs) that operates independently of representation choice, \emph{representation-mediated} robustness (RF, MLP, last-layer BNNs) that depends critically on the input features, and ensemble-based robustness (LightGBM, XGBoost) that benefits from dataset-specific structure. The BNN natural experiment demonstrates that full Bayesian treatment converts a representation-mediated architecture into an inherently robust one ($p < 10^{-7}$), while last-layer treatment provides no benefit.

These mechanisms are linked through uncertainty quantification: among probabilistic models, noise detection ability and robustness are strongly correlated ($\rho \approx 0.90$, $p < 0.01$). NGBoost, which best detects noise in its uncertainty estimates, also resists noise most effectively and produces the best-calibrated intervals. QRF, whose uncertainty is blind to noise, shows the steepest degradation---and is significantly less robust than its deterministic RF counterpart across all datasets tested.

Across all six noise strategies, model rankings were highly concordant ($W = 0.957$), and the relative severity of noise strategies was uniform across architectures---the noise structure, not the model, determines relative damage. However, robustness rankings did not transfer from QM9 to smaller experimental datasets ($r = -0.30$, $p = 0.51$). Critically, whether findings transfer depends on the mechanism: SVM's inherent robustness generalizes to new domains, while XGBoost's dataset-specific robustness does not. The NoiseInject framework released with this work provides tools for benchmarking model robustness through controlled noise injection on arbitrary datasets.

\section*{Declarations}

\paragraph{Competing interests}
The authors declare no competing interests.

% Some journals require declarations to be submitted in a standardised format. Please check the Instructions for Authors of the journal to which you are submitting to see if you need to complete this section. If yes, your manuscript must contain the following sections under the heading `Declarations':

% \begin{itemize}
% \item Funding
% \item Conflict of interest/Competing interests (check journal-specific guidelines for which heading to use)
% \item Ethics approval and consent to participate
% \item Consent for publication
% \item Data availability 
% \item Materials availability
% \item Code availability 
% \item Author contribution
% \end{itemize}

\noindent
% If any of the sections are not relevant to your manuscript, please include the heading and write `Not applicable' for that section. 



\begin{appendices}

\section{Supplementary Tables and Figures}

\begin{table}[h]
\centering
\caption{Supplementary Table S1. Default hyperparameters for all models evaluated in this study.}
\label{tab:supp_s1_hyperparameters}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Model} & \textbf{Hyperparameter} & \textbf{Default Value} \\ 
\midrule
\multicolumn{3}{l}{\textit{Tree-based Models}} \\
\midrule
Random Forest & n\_estimators & 100 \\
              & max\_depth & None \\
              & max\_features & sqrt \\
              & min\_samples\_leaf & 1 \\
              & min\_samples\_split & 2 \\
              & bootstrap & True \\
\addlinespace
Quantile Forest & n\_estimators & 300 \\
                & max\_depth & None \\
                & max\_features & sqrt \\
                & min\_samples\_leaf & 1 \\
                & min\_samples\_split & 2 \\
                & bootstrap & True \\
                & quantile & 0.5 \\
\addlinespace
XGBoost & n\_estimators & 100 \\
        & max\_depth & 6 \\
        & learning\_rate & 0.1 \\
        & subsample & 1.0 \\
        & colsample\_bytree & 1.0 \\
        & colsample\_bylevel & 1.0 \\
        & min\_child\_weight & 1 \\
        & gamma & 0.0 \\
        & reg\_alpha & 0.0 \\
        & reg\_lambda & 1.0 \\
\addlinespace
LightGBM & n\_estimators & 100 \\
         & learning\_rate & 0.1 \\
         & num\_leaves & 31 \\
         & max\_depth & $-1$ (unlimited) \\
         & subsample & 1.0 \\
         & colsample\_bytree & 1.0 \\
         & min\_child\_samples & 20 \\
         & reg\_alpha & 0.0 \\
         & reg\_lambda & 0.0 \\
\addlinespace
NGBoost & learning\_rate & 0.01 \\
        & n\_estimators & 500 \\
        & natural\_gradient & True \\
\midrule
\multicolumn{3}{l}{\textit{Kernel-based Models}} \\
\midrule
SVM & C & 1.0 \\
    & gamma & scale \\
    & kernel & rbf \\
\addlinespace
Gauche (GP) & kernel & Tanimoto \\
            & outputscale & 1.0 \\
            & likelihood\_noise & 0.001 \\
\midrule
\multicolumn{3}{l}{\textit{Neural Networks}} \\
\midrule
DNN & hidden\_size\_1 & 128 \\
    & hidden\_size\_2 & 64 \\
    & activation & ReLU \\
    & dropout & 0.2 \\
    & learning\_rate & 0.001 \\
    & batch\_size & 32 \\
\addlinespace
MLP & hidden\_size & 32 \\
    & num\_hidden\_layers & 2 \\
    & dropout\_rate & 0.2 \\
    & learning\_rate & 0.001 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item[a] Applied to DNN and MLP architectures.
\end{tablenotes}
\end{table}


To validate the ANOVA design, we computed pairwise Spearman rank correlations between all model NDS profiles and between all representation NDS profiles (Supplementary Tables S5 and S6). Models with $\rho > 0.99$ (e.g., conformal wrappers vs. their base models) were excluded from the ANOVA to avoid inflating model degrees of freedom with near-duplicate levels. Representations with $\rho > 0.90$ (SNS vs. ECFP4) were similarly excluded. We also computed ICC(1,1) for all model pairs to assess within-family consistency (Supplementary Table S7).

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_methods_noise_strategies_detailed.png}
\caption{Supplementary Figure S1. Detailed noise strategy distributions at multiple $\sigma$ levels showing the progression of label corruption for each strategy.}
\label{fig:supp_noise_detailed}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig1_supp_ecfp4_overview.png}
\caption{Supplementary Figure S2. Global overview of noise robustness on ECFP4 representation. (A) Performance degradation curves under Gaussian noise. (B) NDS heatmap across model architectures and noise strategies.}
\label{fig:supp_ecfp4_overview}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig3_supp_ecfp4_ranking.png}
\caption{Supplementary Figure S3. NDS heatmap and baseline R$^2$ versus NDS scatter for ECFP4 representation under Gaussian noise.}
\label{fig:supp_ecfp4_ranking}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_full_overview_supp.png}
\caption{Supplementary Figure S4. All model--representation configurations for remaining noise strategies: quantile, heteroscedastic, and value-proportional noise.}
\label{fig:full_overview_supp}
\end{figure}

% TODO [DATA PENDING]: Update validation supplementary figures once all validation jobs complete.
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_strategy.png}
\caption{Supplementary Figure S5. Noise strategy effect on robustness across validation datasets. Mean NDS by strategy and dataset.}
\label{fig:supp_validation_strategy}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_rep_comparison.png}
\caption{Supplementary Figure S6. Representation effect on robustness across validation datasets. Mean NDS by representation and dataset.}
\label{fig:supp_validation_rep}
\end{figure}

% TODO [DATA PENDING]: Per-dataset QM9 correlation figures will update as validation data completes.
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig3_ranking_consistency.png}
\caption{Supplementary Figure S7. Ranking consistency on PDV. (A) NDS heatmap across models and strategies. (B) Baseline R$^2$ versus NDS scatter. (C) Cross-dataset NDS comparison.}
\label{fig:supp_ranking_consistency}
\end{figure}

% Supplementary tables referenced from main text
% TODO: Format these CSVs into proper LaTeX tables or provide as supplementary data files.
\begin{table}[h]
\centering
\caption{Supplementary Table. Configurations excluded from robustness analysis due to baseline R$^2 \leq 0.6$. Available as \texttt{excluded\_configs.csv}.}
\label{tab:excluded_configs}
\small
\textit{63 configurations excluded; see supplementary data file.}
\end{table}

\begin{table}[h]
\centering
\caption{Supplementary Table. NDS values across all model--representation--strategy combinations. Available as \texttt{table2\_supp\_nds\_all\_reps.csv}.}
\label{tab:nds_all_reps}
\small
\textit{Full NDS table; see supplementary data file.}
\end{table}

\begin{table}[h]
\centering
\caption{Supplementary Table. Simple effects analysis: $\eta^2$ for model architecture within each representation level, and vice versa, by noise strategy. Available as \texttt{table1\_supp\_simple\_effects.csv}.}
\label{tab:simple_effects}
\small
\textit{Simple effects analysis; see supplementary data file.}
\end{table}

\begin{table}[h]
\centering
\caption{Supplementary Table. Model rankings at each noise level on PDV (Gaussian strategy), showing how relative model performance changes as noise increases. Available as \texttt{table5\_sigma\_rankings.csv}.}
\label{tab:sigma_rankings}
\small
\textit{Sigma-level rankings; see supplementary data file.}
\end{table}

% An appendix contains supplementary information that is not an essential part of the text itself but which may be helpful in providing a more comprehensive understanding of the research problem or it is information that is too cumbersome to be included in the body of the paper.

%%=============================================%%
%% For submissions to Nature Portfolio Journals %%
%% please use the heading ``Extended Data''.   %%
%%=============================================%%

%%=============================================================%%
%% Sample for another appendix section			       %%
%%=============================================================%%

%% \section{Example of another appendix section}\label{secA2}%
%% Appendices may be used for helpful, supporting or essential material that would otherwise 
%% clutter, break up or be distracting to the text. Appendices can consist of sections, figures, 
%% tables and equations etc.

\end{appendices}
\clearpage

%%===========================================================================================%%
%% If you are submitting to one of the Nature Portfolio journals, using the eJP submission   %%
%% system, please include the references within the manuscript file itself. You may do this  %%
%% by copying the reference list from your .bbl file, paste it into the main manuscript .tex %%
%% file, and delete the associated \verb+\bibliography+ commands.                            %%
%%===========================================================================================%%

\bibliography{sn-bibliography}% common bib file
%% if required, the content of .bbl file can be included here once bbl is generated
%%\input sn-article.bbl


\end{document}
