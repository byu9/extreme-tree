# Nonstationary GEV estimator

This is the companion repository for the paper _Prediction of the Conditional Probability Densities of
Time Interval Extrema with Application to Risk-Sensitive Scheduling_.
https://doi.org/10.48550/arXiv.2506.01358

This repository contains the source code, source datasets, dataset compilation routines, fitting programs, and
visualization scripts. Most programs are in Python but some scripts are written in R and requires the R interpreter.

## Folder Structure

```
.
├── datasets
│   ├── pjm ------------------ contains the PJM dataset and compilation script
│   │   └── raw--------------- contains source data publicly available from PJM data miner 2
│   └── synthetic ------------ contains synthetic dataset compilation script
└── extreme_tree ------------- contains the implementaion of the approach described in our paper
```

## How to Run

The project is developed on Python 3.12 and a relatively recent Python version is required. These are pure python
scripts rather than Jupyter Notebooks, which are meant for interactive demos, while this is a large software project
with multiple custom modules and source lines need to be isolated from results to be properly version-tracked.

To run competitor 2 (GAMLSS), the R interpreter is also required. The websites contain instructions on how to install.

* https://www.python.org
* https://www.r-project.org

### R dependencies

Launch an R terminal and install packages `gamlss`, `gamlssx`, `scoringRules`.
Our approach is implemented in Python. We only use R for the competitor 2 model.

```R
john@Johns-MacBook-Air extreme-tree % which R
/opt/homebrew/bin/R

john@Johns-MacBook-Air extreme-tree % R
> install.packages('gamlss')
> install.packages('gamlssx')
> install.packages('scoringRules')
> q()
```

### Python dependencies

In a terminal, use the following command to install dependencies.

```
(.venv) john@Johns-MacBook-Air extreme-tree % python3 --version
Python 3.13.5

(.venv) john@Johns-MacBook-Air extreme-tree % pip3 install -r requirements.txt
```

## How to replicate results

Running the scripts will create or update the CSV files, which contain the estimated parameters. We included the
result CSV files for convenience. On Linux or Macintosh, launch a terminal and run the scripts directly as follows. Note
the `./` at the beginning.

```
(.venv) john@Johns-MacBook-Air extreme-tree % ./010-run_ours_on_synthetic.py
(.venv) john@Johns-MacBook-Air extreme-tree % ./012-run_competitor2_on_synthetic.R

# Note: We numbered the scripts in the filenames. Press TAB to autocomplete.
```

On MS Windows launch command prompt, run `python3` with the script name

```
C:\WINDOWS\system32> cd C:\Users\JohnYu\Downloads\extreme-tree\
C:\Users\JohnYu\Downloads\extreme-tree> python3 010-run_ours_on_synthetic.py
C:\Users\JohnYu\Downloads\extreme-tree> Rscript 012-run_competitor2_on_synthetic.R
```

If `python3` or `Rscript` cannot be found on MS Windows, the environment variable `Path` is not setup correctly.
Explicitly
calling `python3` or `Rscript` may be required (they are usually installed under `"C:\Program Files\"`)

## Dumping conditional estimator nodes as Tikz diagram

We provided the patch to modify our program to dump out the tree nodes as Tikz diagram. We used this method to create
one of the figures in the paper. Apply the patch as follows

```
(.venv) john@Johns-MacBook-Air extreme-tree % git apply print-fitted-trees-as-latex-diagram.patch 
(.venv) john@Johns-MacBook-Air extreme-tree % ./010-run_ours_on_synthetic.py 
```

The patch will modify our program `extreme_tree/extreme_tree.py` so that it dumps out the node of the first conditional
estimator in the ensemble and exits. The output can then be copy-pasted into Overleaf to construct a LaTeX diagram. The
following is a sample.

```
\node (0x10C803280) [                                                                                                                                                                                  
    tree-node,                                                                                                                                                                                         
    label=above:{\texttt{Node 0x10C803280}},                                                                                                                                                           
    ] {
  $m=1$\\
  $T=0.0902$\\
  $\hat{\mu}=-0.1984$\\
  $\hat{\sigma}=1.3060$\\
  $\hat{\xi}=-0.0616$};
\node (0x10C803400) [
    tree-node,
    label=above:{\texttt{Node 0x10C803400}},
    below left=3em and -2em of 0x10C803280] {
  $m=1$\\
  $T=0.0501$\\
  $\hat{\mu}=0.5637$\\
  $\hat{\sigma}=1.0821$\\
  $\hat{\xi}=0.0047$};
\draw[->, thick] (0x10C803280) -- (0x10C803400) node [midway, left]{$x_{1,n} \leq 1.4985$};
\node (0x10C8033A0) [
    tree-node,
    label=above:{\texttt{Node 0x10C8033A0}},
    below right=3em and -2em of 0x10C803280] {
  $m=1$\\
  $T=0.0353$\\
  $\hat{\mu}=-0.8416$\\
  $\hat{\sigma}=1.0718$\\
  $\hat{\xi}=-0.0071$};
\draw[->, thick] (0x10C803280) -- (0x10C8033A0) node [midway, right]{$x_{1,n} > 1.4985$};
```

To unapply the patch, use the following command. This will revert the patched source file.

```
(.venv) john@Johns-MacBook-Air extreme-tree % git checkout extreme_tree
Updated 1 path from the index
```

## Questions

For questions on the implementation, please email Buyi Yu at byu9@ncsu.edu.
