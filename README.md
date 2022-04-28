# Visually-Aware-Recommendations
 

In this repository we try to experiment and provide some benchmarks for different recommendation algorithms, mainly we are interested in benchmarking the [VBPR](https://arxiv.org/pdf/1510.01784.pdf) & [CausalRec]()https://arxiv.org/pdf/2107.02390.pdf, and compared their performance with other methods such as MM-MF or the most popular approach.

We use two datasets to do this, the [Tradesy](http://jmcauley.ucsd.edu/data/tradesy/) and the [Amazon Clothing](https://nijianmo.github.io/amazon/index.html) datasets, the latter can be segmented across different categories.

For the modeling and evaluation we use the [Cornac Framework](https://cornac.preferred.ai/) to implement the experiments.

## Getting Started

To run the code locally, first clone this repository and install the requirements.

```bash
git clone https://github.com/omartinez182/Visually-Aware-Recommendations.git && cd Visually-Aware-Recommendations
pip install -r requirements.txt
```
