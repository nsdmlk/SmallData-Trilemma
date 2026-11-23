![Small Data Machine Learning](Header.png)

The study examines how model selection, target corruption and interval calibration affect predictions on small tabular datasets. It compares fixed configurations of SmallGBM and SmallMLP with task-specific baselines. [Manuscript (PDF)](Paper.pdf) · [LaTeX source](paper.tex)

## Findings

- At 100 training observations, five-fold cross-validation overstated the gain from selecting XGBoost hyperparameters by 0.063 ROC-AUC on random labels. On the Gaussian task, tuning improved independent-test ROC-AUC by 0.0078 after refitting.
- On a Gaussian mixture with linear log odds, untuned logistic regression had the highest mean ROC-AUC at all seven sizes. On the measured grid, XGBoost-50 first passed SmallGBM in mean ROC-AUC at 1,000 observations, and XGBoost-200 at 2,000. [Gaussian trials](results/revised_figures/gaussian/)
- At 20% random binary label flips, the median within-trial ROC-AUC loss was 0.0210 for SmallGBM and 0.0778 for fixed XGBoost-200. Under Cauchy response corruption, the absolute-error XGBoost regressor attained mean MAE 1.732 at 20%. [Corruption trials](results/revised_figures/corruption/)
- With a 90% target and 38 calibration observations, ordinary split conformal used the 36th sorted residual. At 10 features, its coverage/mean width was 91.5%/5.410 versus 92.7%/5.575 with local weights. At 100 features the corresponding results were 93.2%/8.064 and 94.3%/8.550. Weighted intervals covered 87.4% of high-noise observations at 10 features. Their coverage is an empirical result. [Interval results](results/conformal_uq/20261121-135012-812880/)
- On six public datasets, a linear baseline led on Hepatitis, Breast Cancer and Energy Efficiency. XGBoost led on Ionosphere and Yacht, and SmallGBM on Parkinson's. Results use three repeats of five-fold cross-validation and preserve subject/design groups where applicable. [Scores, predictions and splits](results/real_datasets/)
- A 45-setting width sensitivity check on Iris, Wine and Statlog Heart gave mean accuracy 0.9233 at the existing coefficients (6, 0.7). The best cell on this fixed set of folds scored 0.9288. [Development grid and folds](results/meta_training/20261121-135431-895422/)

## Reproduce the comparisons

Run the notebooks in [benchmarks](benchmarks/) for cross-validation, Gaussian classification, target corruption and network depth. Each records its protocol. From the repository root the following commands reproduce the interval, dataset and width experiments.

```sh
python benchmarks/conformal.py --jobs 2
python benchmarks/real_datasets.py
python benchmarks/alpha_beta_ablation_benchmark.py
```

The classifiers and regressors used SmallGBM 1.5.0, SmallMLP 0.4.0, XGBoost 3.4.1 and scikit-learn 1.9.1. Experiment-specific versions and seeds are recorded in the respective protocol files. For Overleaf, use `paper.tex` as the main file. We verified the included LaTeX project with Tectonic.

## Impact

In a small cohort, reusing the validation score that selected a model can exaggerate its expected gain. The interval experiment illustrates a separate decision risk: an overall coverage near 93% coincided with 87.4% in the high-noise group. Overall and subgroup coverage, interval width and the cost of missed cases all matter when deciding whether another measurement is needed. The clinical dataset results are retrospective and do not establish clinical benefit.

## Citation

If you use this paper or the associated code in your research, please cite it as follows:

```bibtex
@misc{emelyanov2026smalldata,
  title={Small Data Machine Learning: The Trilemma of Low-Sample Tabular Regimes},
  author={Emelyanov, Ilya and Dobrydin, Dmitrii and Hamza, Muhammad and Khan, Malak Abid Ali},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Preprint}
}
```

## Acknowledgements

This research was conducted independently during undergraduate studies at Beijing Institute of Technology (BIT).

## License

**Text and Figures:** This manuscript and all accompanying figures are licensed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).

**Code:** The benchmark scripts and source code in this repository are licensed under the [MIT License](https://opensource.org/licenses/MIT).

The bundled LaTeX style is by George Kour. [Style license](arxiv-style-LICENSE.txt)
