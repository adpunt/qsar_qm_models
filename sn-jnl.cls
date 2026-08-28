% ==============================================================================
% sn-jnl.cls — LOCAL SHIM ONLY. Not a real Springer Nature class.
% Exists solely so this Mac (no sn-jnl install) can pdflatex paper.tex as a
% smoke test. Do NOT include this in any submission. Delete before pushing.
% ==============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/06/01 v0.1 sn-jnl LOCAL SHIM]

% Track the two journal-formatting options that matter for the smoke test
\newif\ifsn@lineno\sn@linenofalse
\newif\ifsn@referee\sn@refereefalse
\DeclareOption{lineno}{\sn@linenotrue}
\DeclareOption{referee}{\sn@refereetrue}
% Swallow every other class option without complaint
\DeclareOption*{}
\ProcessOptions\relax

% Underlying class
\LoadClass[11pt,a4paper]{article}

% Packages it would otherwise pull in
\RequirePackage[margin=2.5cm]{geometry}
\RequirePackage{natbib}
\RequirePackage{lineno}
\RequirePackage{setspace}

% Author / affiliation API used by paper.tex --------------------------------
\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\spfx}[1]{#1}
\newcommand{\sfx}[1]{#1}

% \author[1]{...} and \author*[1]{...} — drop the star/optional, keep the name
\let\sn@orig@author\author
\renewcommand{\author}{\@ifstar{\sn@xauthor}{\sn@xauthor}}
\newcommand{\sn@xauthor}[2][]{\sn@orig@author{#2}}

% \email{...} attaches to most recent \author — render as small footnote text
\newcommand{\email}[1]{\thanks{Email: #1}}

% \affil[1]{...} and \affil*[1]{...} — print as institutional address line
\newcommand{\affil}{\@ifstar{\sn@xaffil}{\sn@xaffil}}
\newcommand{\sn@xaffil}[2][]{\thanks{[#1] #2}}

% Address sub-fields — render inline
\newcommand{\orgdiv}[1]{#1, }
\newcommand{\orgname}[1]{#1, }
\newcommand{\orgaddress}[1]{#1}
\newcommand{\street}[1]{#1, }
\newcommand{\city}[1]{#1, }
\newcommand{\postcode}[1]{#1, }
\newcommand{\country}[1]{#1}

% \title[short]{long} — drop the optional short title
\let\sn@orig@title\title
\renewcommand{\title}[2][]{\sn@orig@title{#2}}

% \abstract{...} as an environment-like command
\renewcommand{\abstract}[1]{%
  \begin{quote}\noindent\textbf{Abstract.\ }#1\end{quote}}

\newcommand{\keywords}[1]{\par\noindent\textbf{Keywords:} #1\par}
\newcommand{\pacs}[2][]{}

% Graphics path so figures resolve when using -output-directory
\AtBeginDocument{\graphicspath{{results/paper_figures/}{./}}}

% Apply journal formatting hooks
\AtBeginDocument{%
  \ifsn@lineno\linenumbers\fi
  \ifsn@referee\doublespacing\fi
}

% Theorem styles used in the file ------------------------------------------
\RequirePackage{amsthm}
\newtheoremstyle{thmstyleone}{}{}{\itshape}{}{\bfseries}{.}{ }{}
\newtheoremstyle{thmstyletwo}{}{}{}{}{\bfseries}{.}{ }{}
\newtheoremstyle{thmstylethree}{}{}{}{}{\bfseries}{.}{ }{}
\theoremstyle{thmstyleone}

% Reference style — let \cite / \citep both work (numbered)
\bibliographystyle{plain}

\endinput
