# Visually-Aware-Recommendations
 

In this repository we run two experiments and provide some benchmarks for different recommendation algorithms, mainly we are interested in benchmarking the [VBPR](https://arxiv.org/pdf/1510.01784.pdf) & [CausalRec](https://arxiv.org/pdf/2107.02390.pdf) recommenders and compare their performance with the more traditional recommenders BPR and MostPop.

We use two datasets to do this, the [Tradesy](http://jmcauley.ucsd.edu/data/tradesy/) and the [Amazon Clothing](https://nijianmo.github.io/amazon/index.html) datasets, the latter can be segmented across different categories.

For the modeling and evaluation we use the [Cornac Framework](https://cornac.preferred.ai/) to implement the experiments.

## Getting Started

1) To run the code locally, first clone this repository and install the requirements.

```bash
git clone https://github.com/omartinez182/Visually-Aware-Recommendations.git && cd Visually-Aware-Recommendations
pip install -r requirements.txt
```
2) Then you can proceed to run the experiment with:

```bash
$ python3 run_experiment.py
```
The default dataset is the Amazon Clothing, but you can pass the dataset as an argument, so if you'd like to run the experiment with the Tradesy dataset you can use the following:

```bash
$ python3 run_experiment.py --dataset tradesy
```
Here's an example of how to execute the script using Google [Colab](https://colab.research.google.com/drive/1tsawUOF5qRAHwYhSEd8MRWHtthel6A3B?usp=sharing), for this make sure that you enable the GPU under ```Runtime``` -> ```Change runtime type``` -> ```Hardware accelerator``` -> ```GPU```.

**NOTE:** Because these datasets are fairly large, sometimes the download process fails, if that happens, try to run the script a few times until it successfully downloads the data.

## Results

The results of the experiment are saved as a log file under:

```data``` -> ```output``` -> ```model_eval```.

**Tradesy VBPR Experiment Results**

<img width="395" alt="image" src="https://user-images.githubusercontent.com/63601717/165853906-7d5fe3ed-79f8-44b4-bb7b-b34d24bcf7e4.png">

**Amazon Clothing VBPR Experiment Results**


## Citations
```
@article{salah2020cornac,
  title={Cornac: A Comparative Framework for Multimodal Recommender Systems},
  author={Salah, Aghiles and Truong, Quoc-Tuan and Lauw, Hady W},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={95},
  pages={1--5},
  year={2020}
}
```