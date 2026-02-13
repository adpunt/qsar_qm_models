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
\abstract{Predictive models for chemical bioactivity are limited by experimental noise and intrinsic biological and chemical variability. Traditional machine learning approaches treat these noisy experimental labels as ground truth and can overfit to the noise itself. Probabilistic modeling, which provides uncertainty quantification, has been used to address some of these issues; however, the evaluation of different probabilistic approaches in terms of comparative effectiveness across different noise conditions and molecular representations remains limited. In this study, we examine noise robustness and uncertainty quantification in a wide range of contexts, comparing the effects of different molecular representations and model architectures under a variety of types of label noise. We injected artificial noise into both clean quantum-chemical datasets and experimental ADME datasets. We demonstrate that while molecular representation explains more variance in overall QSAR performance, model architecture explains more variance in specifically noise robustness. The modeling approaches that showed the highest performance when training on noisy labels were Gaussian processes, followed by both deterministic and probabilistic tree-based approaches. These probabilistic models also provide uncertainty estimates that moderately correlate with prediction error. Model rankings remain stable across different types of artificial noise but show greater variance across different predicted targets. We introduce NoiseInject, an open-source benchmarking framework for evaluating noise robustness across arbitrary datasets, optimized for molecular data but usable on other subjects.}


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

In this study, we evaluated a variety of molecular representations. We used two circular topological fingerprints: the Extended Connectivity Fingerprint with radius 4 (ECFP4) and the \emph{Sort \& Slice} (SNS) fingerprint \citep{Dablander2024}. ECFP4 fingerprints were generated with \texttt{RDKit} \citep{rdkit} using radius $r=2$ and length $d=2048$. Each bit of the fingerprint encodes the hashed identity of one or more circular substructures. One downside of hashed fingerprints is bit collisions, in which chemically distinct substructures map to the same index, resulting effectively in representation-level noise. To address this issue, the SNS fingerprint uses a collision-free alternative to pool ECFP substructures \citep{Dablander2024}. Substructures are sorted by prevalence in the training set, and the top $L$ ($L=1024$) are sliced to create a binary vector. We also used 200-dimensional physicochemical descriptor vectors (PDVs) computed from RDKit molecular descriptors, and various forms of one-hot-encoded (OHE) SMILES strings \citep{Bertoni2020}, generated with \texttt{RDKit} \citep{rdkit}. A random variant of these SMILES was also used, in which the starting point of the atomic sequence is randomized prior to tokenization. We included mol2vec embeddings \citep{jaeger2018mol2vec}, which apply a Word2Vec-inspired approach to molecular substructures. Molecules are decomposed into Morgan substructures at multiple radii, treated as ``words'' in a molecular ``sentence,'' and a pre-trained Word2Vec model produces 300-dimensional embeddings for each substructure. The molecular embedding is the mean of its constituent substructure vectors, yielding a fixed-length, continuous representation that captures substructural context without requiring task-specific training. Finally, we included graph embeddings generated by Molecular Hypergraph Grammar GNNs (MHG-GNN) \citep{kishimoto2023}. MHG-GNNs are a GIN-based autoencoder pretrained on 1.34 million PubChem molecules using $\beta$-VAE loss, producing 1024-dimensional embeddings through iterative message passing. MHG-GNN was originally developed for materials science, although it has demonstrated strong performance on property prediction tasks for polymers, photoresists, and chromophores \citep{kishimoto2023}. 

All representations are precomputed through a Rust–Python pipeline. Rust performs various heavy-computation tasks involved in feature extraction, normalization, and serialization, and outputs the results into memory-mapped binaries (\texttt{.mmap}) for zero-copy reads. Model training, testing, and validation are then done in Python, reading in from the memory-mapped files.

\subsection{Models}
% TODO [REPETITION]: The paragraph below (labels as distributions, probabilistic vs deterministic)
% repeats material from the Introduction. RECOMMENDATION: Keep here in Methods; trim the intro version.
In this work, we used a wide range of model architectures to understand both deterministic and probabilistic learning behaviors across molecular representations and levels of noise. One key issue with QSAR models is the standard ML assumption that labels represent true values, as most chemical data sets lack sufficient replications to provide a robust statistical representation of the underlying distributions \citep{Kolmar2021}. A single measurement, or even multiple measurements for a given label, will not guarantee that this label accurately reflects the population mean \citep{Kolmar2021}. Most ML models rely on the assumption that the training data is, in fact, an accurate reflection of the population. Therefore, they treat these potentially noisy labels as discrete quantities rather than distributions, which opens the door to overfitting to noise and giving unduly optimistic assessments of the models. However, probabilistic methods such as Gaussian processes (GPs) and Bayesian neural networks (BNNs) do not make this assumption \citep{Kolmar2021}.

Across all models, hyperparameter tuning was performed using Optuna \citep{akiba2019optuna} Bayesian optimization (Tree-structured Parzen Estimator). The tuned hyperparameters were compared against conventional default values for each architecture, and the configuration with better validation performance was retained as the fixed parameter set for all subsequent experiments.

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

To ensure a valid fully crossed design, we curated the set of ANOVA factor levels by removing redundant or near-duplicate levels. Pairwise Spearman rank correlations were computed between all model NDS profiles and between all representation NDS profiles across noise strategies. Models whose NDS profiles correlated at $\rho > 0.99$ with another model were excluded: this removed conformal prediction wrappers (conformal RF, conformal QRF, conformal DNN), which produced near-identical robustness profiles to their base models, and quantile regression forests (QRF), which were highly redundant with RF. Among representations, SNS and ECFP4 correlated at $\rho > 0.90$, and SNS was excluded to avoid inflating representation degrees of freedom. Randomized SMILES was excluded due to incomplete coverage across models. We also computed intraclass correlation coefficients ICC(1,1) for all model pairs to assess within-family consistency; ICC(1,1) measures the proportion of total variance attributable to between-subject (i.e., between-configuration) differences, with values near 1.0 indicating that two models rank configurations almost identically. The full redundancy and ICC tables are reported in Supplementary Tables S5--S7. After exclusion, the ANOVA retained all remaining model architectures and five representations (ECFP4, PDV, SMILES, mol2vec, and MHGGNN), yielding a fully crossed design.

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

% TODO: in all figures, make sure task (QM9) is mentioned

% TODO: Update NDS values, ANOVA η², and tables once hetero/valprop data completes for all models.
\subsection{Model Robustness Under Noise}

We evaluated all model architectures under increasing artificial noise on the PDV representation across six noise strategies (Figure~\ref{fig:global_overview}). Performance degrades approximately linearly with noise level for all models, but the rate of degradation varies substantially across architectures.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig1_global_overview.png}
    \caption{\textbf{Global overview of noise robustness.} (A) Performance degradation curves (PDV, Gaussian noise) showing R$^2$ versus noise level $\sigma$ for representative models. (B) NDS heatmap across all model architectures and noise strategies on PDV; darker cells indicate steeper degradation.}
    \label{fig:global_overview}
\end{figure}

On the Gaussian strategy, BNN variants showed the shallowest degradation: var-BNN achieved NDS $= -0.282$ and full-BNN achieved NDS $= -0.338$, outperforming all deterministic architectures. NGBoost (NDS $= -0.310$) and SVM (NDS $= -0.337$) also demonstrated strong robustness. At the other end, QRF was the least robust model (NDS $= -0.397$), performing worse than its deterministic counterpart RF (NDS $= -0.364$). LightGBM (NDS $= -0.349$) and XGBoost (NDS $= -0.349$) showed similar intermediate robustness.

Averaging across all six strategies, NGBoost achieved the best mean robustness (mean NDS $= -0.323$), followed by XGBoost ($-0.328$), SVM ($-0.347$), and var-BNN ($-0.356$). QRF ($-0.407$) and MLP ($-0.404$) were the least robust overall. However, the cross-strategy average masks substantial variation: model standard deviations in NDS ranged from $0.21$ to $0.28$, indicating that robustness rankings are strategy-dependent.

The consistency of model rankings across noise strategies was assessed using Kendall's coefficient of concordance ($W = 0.873$, $\chi^2 = 62.84$, $p < 0.001$), indicating strong agreement. Despite this overall concordance, individual models showed notable strategy-specific behavior. GP (Gauche) exhibited the widest NDS range of any model: nearly unaffected by outlier noise (NDS $= -0.12$) but severely degraded by threshold noise (NDS $= -0.69$). This variability suggests that GP's robustness is highly dependent on the structure of the noise, likely because threshold noise, which systematically corrupts all labels above a cutoff, violates the GP's stationarity assumptions.

To ensure meaningful robustness comparisons, we restricted analysis to configurations achieving baseline R$^2 > 0.6$. Configurations with poor baseline performance can exhibit misleadingly shallow degradation slopes simply because there is less performance to lose; a model that starts at R$^2 = 0.4$ cannot degrade as steeply as one starting at R$^2 = 0.85$. This threshold excluded 75 configurations, of which 14 were marginal ($0.5 \leq R^2 < 0.6$). The full exclusion list is in Supplementary Table~\ref{tab:excluded_configs}.

\subsection{Strategy-Specific Patterns}

The six noise strategies produced markedly different degradation profiles (Figure~\ref{fig:global_overview}B, Table~\ref{tab:anova_decomposition}). Outlier noise was the mildest: all models showed NDS values near zero (range: $-0.10$ to $-0.20$), consistent with the ability of ensemble methods to isolate outlying observations. Heteroscedastic noise was similarly benign for models with complete data (NDS range: $-0.10$ to $-0.21$), suggesting that value-dependent noise variance is relatively easy to accommodate.

Gaussian noise and quantile noise produced moderate degradation, with NDS values clustering around $-0.31$ to $-0.40$ and $-0.27$ to $-0.35$ respectively. Quantile noise, which applies elevated corruption to samples in the tails of the target distribution, was somewhat milder than Gaussian noise for most models, possibly because the corrupted region is smaller.

Threshold noise was the most destructive strategy across all architectures, with NDS values ranging from $-0.66$ to $-0.79$. Threshold noise deterministically corrupts all labels above a property cutoff, effectively introducing a systematic bias that models cannot easily learn around. Value-proportional noise was similarly severe (NDS range: $-0.57$ to $-0.65$ for models with complete data), as it scales corruption with the magnitude of the target value, disproportionately affecting the most informative high-value samples.

The ANOVA decomposition revealed that the relative importance of model architecture versus representation depends strongly on the noise strategy (Table~\ref{tab:anova_decomposition}, Figure~\ref{fig:anova_decomposition}). For performance ($R^2$ at $\sigma = 0.3$), representation explained 24--34\% of variance on most strategies, while model explained 21--24\%. However, on heteroscedastic and value-proportional noise, model architecture became the dominant factor ($\eta^2 = 52.6\%$ and $50.7\%$ respectively), suggesting that these structured noise types expose differences in how architectures handle non-uniform error distributions.

For robustness (NDS), the pattern was even more strategy-dependent. Model architecture dominated on Gaussian ($\eta^2 = 43.8\%$), threshold ($33.8\%$), and heteroscedastic ($48.7\%$) noise. But on outlier and value-proportional noise, neither model nor representation explained substantial variance ($\eta^2 < 9\%$ for both factors), with the residual and interaction terms absorbing most of the variation. This indicates that for these strategies, robustness is determined by configuration-specific interactions rather than by either factor alone.

The strategy sensitivity ratio, defined as a model's NDS on a given strategy divided by its Gaussian NDS, quantifies the relative severity of each noise type. Remarkably, these ratios were uniform across all model architectures: threshold noise was approximately $1.9$--$2.0\times$ as damaging as Gaussian noise for every model, quantile noise was $0.81$--$0.93\times$, and outlier noise was $0.29$--$0.45\times$ (Table~\ref{tab:strategy_sensitivity}). This uniformity suggests that the relative severity of noise strategies is a property of the noise structure itself rather than of model-specific vulnerabilities, and that different architectures respond proportionally to the same noise perturbations.

\begin{table}[htbp]
\centering
\caption{\textbf{Strategy sensitivity ratios relative to Gaussian noise.} Each value represents a model's NDS on the given strategy divided by its Gaussian NDS; values $> 1$ indicate greater damage than Gaussian, $< 1$ indicate less. Ratios are strikingly uniform across architectures, indicating that relative strategy severity is noise-intrinsic.}
\label{tab:strategy_sensitivity}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Outlier} & \textbf{Quantile} & \textbf{Threshold} & \textbf{Hetero.} & \textbf{Val.-Prop.} \\
\midrule
NGBoost  & 0.33 & 0.87 & 1.97 & ---  & ---  \\
XGBoost  & 0.35 & 0.43 & 1.98 & ---  & ---  \\
SVM      & 0.30 & 0.84 & 1.98 & ---  & ---  \\
RF       & 0.32 & 0.86 & 2.00 & ---  & ---  \\
LightGBM & 0.32 & 0.86 & 1.98 & 0.55 & 1.75 \\
GP       & 0.34 & 0.87 & 1.91 & 0.28 & 1.58 \\
DNN      & 0.37 & 0.91 & 1.93 & 0.55 & 1.74 \\
MLP      & 0.34 & 0.92 & 1.92 & 0.55 & 1.74 \\
\bottomrule
\end{tabular}
\end{table}

% TODO: make sure that if a, b, c labels are lowercase, they're referred to as lowercase

\begin{table}[htbp]
\centering
\caption{\textbf{ANOVA variance decomposition by noise strategy.} $\eta^2$ values (\%) for model architecture and molecular representation effects on predictive performance (R$^2$ at $\sigma=0.3$) and noise robustness (NDS). Interaction and residual terms account for the remaining variance. Representation dominates performance on most strategies, while model architecture dominates robustness on Gaussian, threshold, and heteroscedastic noise. On outlier and value-proportional noise, neither factor explains substantial robustness variance.}
\label{tab:anova_decomposition}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Performance ($\eta^2$, \%)}} & \multicolumn{2}{c}{\textbf{Robustness ($\eta^2$, \%)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Strategy} & \textbf{Model} & \textbf{Rep} & \textbf{Model} & \textbf{Rep} \\
\midrule
Gaussian        & 22.4 & 31.5 & 43.8 & 9.4 \\
Quantile        & 23.4 & 33.0 & 12.4 & 2.3 \\
Threshold       & 23.3 & 33.8 & 33.8 & 10.5 \\
Heteroscedastic & 52.6 & 23.5 & 48.7 & 12.8 \\
Value-prop.     & 50.7 & 23.9 &  8.8 & 0.2 \\
Outlier         & 21.9 & 32.6 &  1.6 & 0.2 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig2_anova_decomposition.png}
    \caption{\textbf{ANOVA variance decomposition by noise strategy.} Variance explained ($\eta^2$, \%) for (A) predictive performance (R$^2$ at $\sigma=0.3$) and (B) noise robustness (NDS), decomposed into model architecture, molecular representation, and interaction effects for each noise strategy.}
    \label{fig:anova_decomposition}
\end{figure}

\begin{table}[htbp]
\centering
\caption{\textbf{Noise Degradation Slope by model on PDV.} Models ranked by mean $|$NDS$|$ across strategies with complete data. Lower $|$NDS$|$ indicates greater noise robustness. Only the ANOVA-included models are shown; conformal wrappers and QRF are excluded per the redundancy analysis. Heteroscedastic and value-proportional columns are available only for models with complete data.}
\label{tab:nds_ranking}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Gaussian} & \textbf{Outlier} & \textbf{Quantile} & \textbf{Threshold} & \textbf{Mean} & \textbf{STD} \\
\midrule
NGBoost             & $-0.310$ & $-0.101$ & $-0.270$ & $-0.611$ & $-0.323$ & 0.212 \\
XGBoost             & $-0.349$ & $-0.122$ & $-0.150$ & $-0.693$ & $-0.328$ & 0.263 \\
SVM                 & $-0.337$ & $-0.102$ & $-0.282$ & $-0.667$ & $-0.347$ & 0.236 \\
Var-BNN             & $-0.282$ & $-0.127$ & $-0.328$ & $-0.688$ & $-0.356$ & 0.237 \\
GP (Gauche)         & $-0.362$ & $-0.123$ & $-0.316$ & $-0.691$ & $-0.361$ & 0.236 \\
Full-BNN            & $-0.338$ & $-0.128$ & $-0.299$ & $-0.657$ & $-0.367$ & 0.213 \\
LightGBM            & $-0.349$ & $-0.113$ & $-0.299$ & $-0.689$ & $-0.375$ & 0.229 \\
Last-BNN            & $-0.358$ & $-0.134$ & $-0.327$ & $-0.693$ & $-0.378$ & 0.232 \\
Flexible DNN        & $-0.358$ & $-0.133$ & $-0.327$ & $-0.699$ & $-0.379$ & 0.235 \\
RF                  & $-0.364$ & $-0.117$ & $-0.313$ & $-0.729$ & $-0.381$ & 0.256 \\
MLP-BNN (Var)       & $-0.386$ & $-0.142$ & $-0.336$ & $-0.695$ & $-0.390$ & 0.229 \\
DNN                 & $-0.361$ & $-0.134$ & $-0.328$ & $-0.699$ & $-0.392$ & 0.227 \\
MLP-BNN (Last)      & $-0.386$ & $-0.137$ & $-0.345$ & $-0.702$ & $-0.393$ & 0.233 \\
Flex.\ DNN 256-128-64 & $-0.369$ & $-0.203$ & $-0.342$ & $-0.699$ & $-0.403$ & 0.210 \\
MLP                 & $-0.374$ & $-0.126$ & $-0.346$ & $-0.720$ & $-0.404$ & 0.237 \\
QRF                 & $-0.397$ & $-0.116$ & $-0.322$ & $-0.794$ & $-0.407$ & 0.284 \\
\bottomrule
\end{tabular}
\end{table}

Despite the variation in absolute NDS values across strategies, model rankings remained strongly consistent. Kendall's coefficient of concordance yielded $W = 0.873$ ($\chi^2 = 62.84$, $p < 0.001$), indicating that the most robust models under Gaussian noise tend to be the most robust under threshold, quantile, and other strategies as well. The NDS ranking table (Table~\ref{tab:nds_ranking}) illustrates this visually: NGBoost and XGBoost rank near the top across all strategy columns, while MLP and QRF consistently rank near the bottom. The main exception is outlier noise, where the uniformly mild degradation compresses rankings and makes models harder to distinguish. Baseline performance (R$^2$ at $\sigma = 0$) does not strongly predict noise robustness: models with similar clean-data performance can exhibit markedly different degradation slopes, as seen in Figure~\ref{fig:full_overview}. This underscores that robustness is a distinct property from accuracy.

\subsection{Representation--Model Interactions}

While model architecture is the primary driver of noise robustness on most strategies, it cannot be fully decoupled from representation choice. The interaction and residual terms in the ANOVA absorb 40--46\% of performance variance and even more for robustness: up to 85\% on quantile, 91\% on value-proportional, and 98\% on outlier noise (Table~\ref{tab:anova_decomposition}). On these strategies, knowing the model or representation alone is essentially uninformative---robustness is determined by the specific model--representation pairing.

Figure~\ref{fig:full_overview} shows all model--representation configurations plotted by baseline R$^2$ versus NDS for three representative strategies spanning the severity tiers: Gaussian (moderate), outlier (mild), and threshold (severe). Configurations excluded from the ANOVA (conformal wrappers, QRF, SNS, and randomized SMILES) are included to show the full experimental landscape. For each model, the vertical spread of points across representations shows how much robustness depends on the choice of representation. The complete NDS values for all model--representation--strategy combinations are provided in Supplementary Table~\ref{tab:nds_all_reps}. Configurations from the remaining three strategies are shown in Supplementary Figure~\ref{fig:full_overview_supp}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig_full_overview.png}
    \caption{\textbf{All model--representation configurations across noise severity tiers.} Each point is one model--representation configuration; marker shape indicates representation, color indicates model. Panels show (A) Gaussian, (B) outlier, and (C) threshold noise strategies, spanning the mild-to-severe range.}
    \label{fig:full_overview}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig_interaction.png}
    \caption{\textbf{Representation--model interaction effects.} (A) NDS heatmap showing model robustness across representations under Gaussian noise. (B) NDS on PDV versus ECFP4 for each model under Gaussian noise, with Spearman correlation.}
    \label{fig:interaction}
\end{figure}

Figure~\ref{fig:interaction}A shows how model rankings change across representations under Gaussian noise. While cross-strategy rankings are highly concordant ($W = 0.873$), cross-representation rankings show frequent crossings: a model that ranks first on one representation may rank last on another. The NDS scatter comparing PDV and ECFP4 (Figure~\ref{fig:interaction}B) yields a weak, non-significant correlation (Spearman $\rho = 0.40$, $p = 0.069$), confirming that robustness rankings do not transfer across representations.

Simple effects analysis (Supplementary Table~\ref{tab:simple_effects}) reveals why. Some representations amplify model differences: within ECFP4, model architecture explains $74.3\%$ of robustness variance under Gaussian noise, and within SMILES, $90.7\%$. Other representations compress them: within PDV, model explains only $38.4\%$, and within mol2vec, $42.4\%$. From the model side, some architectures are highly sensitive to which representation they receive---representation explains $95.7\%$ of NGBoost's robustness variance---while others are largely invariant, with SVM ($11.8\%$) and full-BNN ($15.3\%$) showing similar robustness regardless of input representation.

The practical implication is that model architecture is the main driver of noise robustness, but it cannot be entirely decoupled from representation selection. Finding the right model paired with the right type of molecular data produces a more robust system than either choice in isolation. However, predicting which pairings will excel is not straightforward from first principles---it must be tested empirically, which is exactly the type of systematic benchmarking that the NoiseInject framework enables.

% TODO [HIGHLIGHT]: Add discussion of strategy-specific model/rep patterns:
% - GP (Gauche) uniquely vulnerable to threshold noise (stationarity violation) but nearly immune to outlier
% - BNN variants have best relative showing under Gaussian (var-BNN NDS=-0.282), less advantage elsewhere
% - SMILES amplifies model differences (90.7% robustness variance from model); wrong model + SMILES = severe penalty
% - PDV compresses/equalizes model differences — more forgiving of model choice
% - NGBoost most rep-sensitive (95.7%), SVM/full-BNN are rep-invariant
% - Hetero + valprop flip the ANOVA: model dominates performance (50-52%), unlike other strategies where rep dominates
% - Quantile + outlier: robustness almost entirely from model-rep pairing, not either factor alone
% Could be a dedicated paragraph here, in Discussion, or both.

% TODO [BLOCKED]: Bayesian Transformations section needs updating once DNN BNN results complete.
% - Fig 4 (DNN family): only DNN has data; BNN-Full/Last/Var bars are empty
% - Fig 5 (MLP + RF): partially useful, MLP family works, RF vs QRF heteroscedastic empty
% - Figure references still point to old skeleton names (figure3_deterministic_vs_probabilistic.png,
%   figure4_bayesian_transformations.png) — remap to fig4_dnn_family.png and fig5_mlp_rf_comparison.png
% - Table 3 Wilcoxon p-values are real and can stay, but DNN BNN rows are missing
% - Once data arrives: regenerate figures, remap references, review numbers
\subsubsection{Bayesian Transformations and Probabilistic Approaches}

One goal of this research was to evaluate whether Bayesian transformations on neural networks improve robustness to label noise. % TODO [DATA PENDING]: Update NDS numbers and p-values once all BNN ANOVA data completes.
Aggregating across all configurations, we found no significant overall difference between deterministic and probabilistic approaches (Mann-Whitney U, $p = 0.841$). Deterministic models achieved mean $|$NDS$|$ of 0.403 ($n=43$) versus 0.403 for probabilistic models ($n=74$). However, architecture-specific comparisons reveal a more nuanced picture (Figure~\ref{fig:dnn_family}).

% TODO [BLOCKED]: fig4_dnn_family.png is generated but may change once BNN ANOVA data completes.
% Update numbers and review figure after all DNN BNN jobs finish.
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig4_dnn_family.png}
    \caption{\textbf{DNN family comparison.} R$^2$ versus noise level $\sigma$ for the DNN and its BNN variants (full, last-layer, variational) across noise strategies.}
    \label{fig:dnn_family}
\end{figure}

For random forests, the QRF variant shows no significant difference from its deterministic counterpart. Mann-Whitney U tests comparing RF versus QRF across representations showed no significant difference in noise robustness ($p = 0.31$), with RF achieving mean $|$NDS$|$ of 0.326 versus 0.354 for QRF, as seen in Figure~\ref{fig:mlp_rf_comparison}.

% TODO [BLOCKED]: fig5_mlp_rf_comparison.png is generated but may change once MLP BNN and RF/QRF
% hetero/valprop ANOVA data completes. Update numbers and review figure after jobs finish.
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{fig5_mlp_rf_comparison.png}
    \caption{\textbf{MLP and RF family comparison.} R$^2$ versus noise level $\sigma$ for the MLP and its BNN variants, and for RF versus QRF, across noise strategies.}
    \label{fig:mlp_rf_comparison}
\end{figure}

For NNs, Bayesian transformations had mixed effects on noise robustness, often dependent on the representation. MLP-BNN-Full showed significantly improved noise robustness compared to deterministic MLP ($p = 0.038$), with mean $|$NDS$|$ of 0.376 versus 0.489. However, last-layer (MLP-BNN-Last) and variational (MLP-BNN-Var) transformations showed no significant improvement ($p = 0.93$ and $p = 0.66$ respectively). Table~\ref{tab:nn_transforms} shows that for vector-based representations (PDV, ECFP4), BNN variants provide minimal improvement over deterministic MLP. However, on SMILES variants, MLP-BNN-Full substantially improves noise robustness.

\begin{table}[htbp]
\centering
\caption{Neural network Bayesian transformation effects by representation.}
\label{tab:nn_transforms}
\small
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Representation} & \textbf{Baseline R$^2$} & \textbf{MLP NDS} & \textbf{BNN-Full NDS} & \textbf{NDS Improvement} \\
\midrule
MLP & ECFP4 & 0.790 & $-0.465$ & $-0.387$ & +0.079 \\
MLP & PDV & 0.828 & $-0.381$ & --- & --- \\
MLP & SNS & 0.813 & $-0.510$ & $-0.398$ & +0.113 \\
MLP & SMILES & 0.728 & $-0.577$ & $-0.371$ & +0.206 \\
MLP & R-SMILES & 0.675 & $-0.529$ & $-0.348$ & +0.180 \\
\bottomrule
\end{tabular}
\end{table}

The pattern suggests that Bayesian transformations are most beneficial when the base model struggles with a representation. Long, string-based representations like SMILES present challenges for NNs in the form of variable length and complex syntax. When we add a full Bayesian transformation to an NN, it appears to regularize learning in this case. However, on representations which NNs typically excel on, the additional complexity seems to provide minimal benefits. 

\subsubsection{Uncertainty Quantification in the Presence of Noise}

We evaluated the uncertainty estimates produced by probabilistic models under increasing label noise, focusing on four questions: (1) do uncertainty estimates correlate with actual prediction error? (2) do models detect when noise has been injected? (3) how well-calibrated are the predicted intervals? and (4) for models with aleatoric/epistemic decomposition, does the correct component respond to noise?

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
\caption{Uncertainty quantification under increasing label noise (ECFP4, Gaussian strategy). (A) Calibration: binned predicted uncertainty vs.\ actual error. Points on the diagonal indicate perfect calibration. (B) Mean predicted uncertainty as a function of injected noise level $\sigma$. Models whose uncertainty grows with $\sigma$ are detecting added noise; flat lines indicate noise-insensitive estimates. (C) Aleatoric vs.\ epistemic uncertainty decomposition across noise levels. Solid lines show aleatoric uncertainty; dashed lines show epistemic. Ideally, aleatoric uncertainty should grow with noise while epistemic remains stable.}
\label{fig:uncertainty_combined}
\end{figure}

Table~\ref{tab:uncertainty_metrics} summarizes the key uncertainty metrics across all probabilistic models. NGBoost and QRF achieve the strongest uncertainty-error correlations, meaning their predicted uncertainty is most informative about where prediction errors are large. Conformal prediction methods show the highest uncertainty-noise correlations by construction, since their intervals are calibrated to cover a fixed proportion of data. NGBoost also tracks noise well, while QRF---despite strong error correlation---shows weak noise detection, suggesting its uncertainty reflects model-specific variance rather than data quality.

Figure~\ref{fig:uncertainty_combined}A reveals calibration differences: NGBoost's curve tracks closest to the diagonal, indicating well-calibrated intervals. QRF tends to overestimate uncertainty in the lower range but converges at higher predicted values. BNN variants show inconsistent calibration, with some substantially underestimating uncertainty.

The noise-tracking behavior in Figure~\ref{fig:uncertainty_combined}B separates models into two groups. NGBoost and conformal methods increase their predicted uncertainty roughly in proportion to $\sigma$, correctly signaling degraded data quality. BNN variants remain largely flat, failing to detect the added noise. GP (Gauche) shows an intermediate response that varies by representation.

Figure~\ref{fig:uncertainty_combined}C shows the aleatoric--epistemic decomposition for models that support it. In an ideal scenario, increased label noise should raise aleatoric uncertainty (reflecting data noise) while epistemic uncertainty (reflecting model limitations) stays constant. Instead, GP's epistemic uncertainty grows with $\sigma$, indicating that the model misattributes data noise to model uncertainty. This misattribution is most pronounced on string-based representations (SMILES, SNS), where GP already has higher epistemic uncertainty at baseline.

Overall, NGBoost produces the most useful uncertainty estimates under noisy conditions: well-correlated with error, responsive to noise level, and well-calibrated. QRF offers strong error correlation but poor noise detection. BNNs, despite their theoretical foundation in Bayesian inference, produce uncertainty estimates that are neither well-correlated with error nor responsive to noise. Supplementary figures show these patterns hold across alternative representations and noise strategies.

% TODO [EXPAND]: Expand uncertainty section with strategy-specific and rep-specific patterns:
% - Do uncertainty-error correlations hold across all 6 noise strategies, or just Gaussian?
% - Is there any strategy where BNN uncertainty actually works?
% - Are QRF's uncertainty patterns representation-dependent?
% - Does GP's epistemic misattribution persist across strategies or is it strategy-specific?
% - Which models' uncertainty is most useful for identifying noisy predictions (practical filtering)?
% Data source: table4_supp_uncertainty_by_strategy_rep.csv

\subsubsection{Generalization Across Noise Injection Strategies}
The majority of experiments were conducted with artificial noise drawn from a Gaussian distribution and randomly distributed across all labels, with the HOMO-LUMO gap as the prediction target. We now assess how our research generalize to alternative noise injection strategies representing different data quality issues.

We evaluated six artificial noise injection strategies designed to mimic various experimental data quality scenarios. Model performance varied substantially across noise strategies (Table~\ref{tab:phase5_noise_strategy}). We clustered the strategies into three tiers based on mean retention percentage: \textbf{mild} (outlier: 2.5\%, heteroscedastic: 6.6\% degradation), \textbf{moderate} (quantile-based: 11.4\%, legacy Gaussian: 19.0\%), and \textbf{severe} (value-proportional: 34.1\%, threshold: 42.2\%).

% NOTE: Noise strategy generalization is now covered by fig1_global_overview.png (Panel B: NDS heatmap
% across strategies) and fig2_anova_decomposition.png (η² breakdown by strategy). Separate figure
% removed to avoid redundancy with Strategy-Specific Patterns section.

\begin{table}[htbp]
\centering
\caption{Robustness performance by noise corruption strategy.}
\label{tab:phase5_noise_strategy}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Noise} & \textbf{Mean} & \textbf{Mean} & \textbf{Std} & \textbf{Mean} \\
\textbf{Strategy} & \textbf{Baseline $R^2$} & \textbf{Retention \%} & \textbf{Retention \%} & \textbf{$|NDS|$} \\
\midrule
Outlier & 0.778 & 97.5 & 0.3 & 0.099 \\
Heteroscedastic & 0.776 & 93.4 & 0.7 & 0.194 \\
Quantile-based & 0.785 & 88.6 & 1.5 & 0.286 \\
Legacy (Gaussian) & 0.831 & 81.0 & 3.3 & 0.362 \\
Value-proportional & 0.785 & 65.9 & 3.6 & 0.612 \\
Threshold & 0.785 & 57.8 & 4.7 & 0.675 \\
\bottomrule
\end{tabular}
\end{table}

These results indicate that the type of label noise has a strong impact on a model's ability to learn in the presence of noise. Outlier corruption was the least damaging, since many models have built-in techniques for identifying and ignoring outliers. For example, tree-based approaches isolate outliers into leaf nodes. Similarly, heteroscedastic noise, which scales with property magnitude, proved relatively benign as models could identify and adapt to the structured noise pattern. Threshold noise on the other hand, which deterministically corrupts all labels above a property cutoff, causes the greatest disruption in predictive performance. 

Gaussian noise is conventionally used to model label noise. However, this misses the full picture. Evaluating performance across different types of noise produces a clearer understanding of how real-world noise would be handled. The heatmap analysis (Figure~\ref{fig:global_overview}B) reveals that while absolute performance varies considerably across modeling techniques, relative model rankings remain consistent. NGBoost configurations achieve the highest retention across all noise types, followed by RF, then DNN.

Despite the variation in absolute performance degradation, relative model rankings remained stable across all six noise strategies (Figure~\ref{fig:global_overview}B).

% TODO [CATASTROPHIC FAILURES]: Address and discuss catastrophic failures:
% - DNN+PDV on hERG-Ki: NDS = -1094 (model collapse under noise, other reps/datasets fine)
% - XGBoost on Caco2_Efflux: NDS -1.1 to -1.4 across all reps (dataset-specific weakness)
% - Caco2_Efflux is universally the hardest dataset for all models
% - Validation failures: filtered by baseline R^2 < 0.6 threshold
% - All exclusions must be explicitly noted in paper text (not silently removed).
%
% TODO [PAPER NOTE]: Add a brief note in Methods or Results:
%   "N training iterations were excluded due to catastrophic training failures
%   (R^2 < -0.5), primarily affecting DNN on mol2vec (see Supplementary Table X).
%   These are attributed to DNN training instability on lower-dimensional
%   representations and do not affect conclusions."
\subsubsection{Generalization Across Prediction Targets}

% TODO [DATA PENDING]: Validation jobs (NGBoost, SVM, LightGBM) may still be running.
% Update text and figures once all validation data is available. Numbers below are preliminary.
To assess whether noise robustness results generalize beyond the HOMO-LUMO gap on QM9, we evaluated model configurations across three additional experimentally-derived molecular property datasets: LogD (lipophilicity) from OpenADMET, Caco-2 efflux permeability from OpenADMET, and hERG-Ki (cardiac ion channel binding affinity, regression) from ChEMBL. All three are regression targets, and all experiments used the NoiseInject framework with noise levels $\sigma \in \{0, 0.1, \ldots, 1.0\}$ across all six noise strategies. These datasets represent realistic ADME property prediction scenarios where experimental noise is inherently present.

Figure~\ref{fig:validation_overview} shows NDS heatmaps for each validation dataset, broken down by model and noise strategy. Configurations with $|$NDS$| > 2.0$ were filtered as artifacts (shown as ``N/A''), and missing model--strategy combinations are indicated. The same baseline R$^2 > 0.6$ threshold was applied to exclude configurations where poor clean-data performance would produce misleading robustness estimates.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_overview.png}
\caption{\textbf{Validation dataset NDS heatmaps.} NDS by model and noise strategy for each external dataset (LogD, Caco-2 efflux, hERG-Ki). Black cells with ``missing'' indicate configurations not yet run; ``N/A'' indicates filtered extreme values.}
\label{fig:validation_overview}
\end{figure}

To assess whether robustness rankings transfer from QM9 to external datasets, we compared per-model mean NDS on QM9 against mean NDS on each external dataset (Figure~\ref{fig:validation_qm9_transferability}), and examined the per-model robustness profile across datasets (Figure~\ref{fig:validation_model_comparison}).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{fig_validation_qm9_transferability.png}
\caption{\textbf{Robustness transferability from QM9 to external datasets.} Per-model mean NDS on QM9 versus mean NDS across all three external datasets, with Pearson correlation and linear trend line.}
\label{fig:validation_qm9_transferability}
\end{figure}

% TODO [DATA PENDING]: Update per-dataset correlation figures and model comparison after
% NGBoost, SVM, LightGBM validation data completes.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_validation_model_comparison.png}
\caption{\textbf{Model robustness across validation datasets.} Mean NDS by model for each external dataset, ordered by overall mean NDS.}
\label{fig:validation_model_comparison}
\end{figure}

\section{Conclusion}\label{sec13}
This study demonstrates that while molecular representation has a stronger impact on QSAR model performance than the choice of model architecture, the roles are reversed for noise robustness; once noise is introduced into the labels, the choice of model architecture becomes more important. We found that tree-based methods and GPs maintained strong performance under label corruption. NGBoost emerged as a particularly strong choice despite its relatively poor performance in clean data. This may be because NGBoost’s probabilistic formulation captures the error structure more effectively than deterministic gradient boosting. Although the model architecture was the primary factor in noise robustness, molecular representation also had an impact, and PDV and SNS fingerprints showed the highest retention under noise.

Probabilistic and deterministic methods demonstrated comparable noise robustness, though probabilistic approaches provide uncertainty estimates that correlate only moderately with prediction error. This is useful information about reliability that can be extracted even when calibration is poor and used in future noise-mitigation research. We show that relative noise-robustness patterns across model architectures and representations remain consistent across different strategies for injecting artificial noise and targets, despite overall predictive performance varying widely. 

The methods presented here enable the identification of noise-robust QSAR molecular representation and model configurations. The NoiseInject framework released with this work provides tools for benchmarking model robustness through controlled noise injection on arbitrary datasets. Future work should investigate whether uncertainty estimates can be leveraged for noise detection and selective data removal to improve model performance on noisy datasets.

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
