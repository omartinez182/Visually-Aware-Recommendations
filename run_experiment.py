import argparse
import cornac
from cornac.datasets import amazon_clothing
from cornac.datasets import tradesy
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from cornac.models import MostPop, BPR, VBPR, CausalRec


def create_parser():
    """
    Function to map dataset names to function objects.
    """
    parser = argparse.ArgumentParser(description='add dataset')
    parser.add_argument('--dataset', help='pass dataset name to be used', 
          choices=dict_names, default=amazon_clothing)
    return parser

dict_names = {'amazon_clothing': amazon_clothing, 'tradesy': tradesy} 

def load_data(dataset):
    """
    Function to load the data from the specified arg.
    """
    fn = dict_names[dataset]
    return fn()

def main():
    """
    Experiment to compare the performance of the MostPop, 
    BPR, VBPR, and CausalRec recommenders on either the Tradesy or
    the Amazon Clothing datasets.
    """
    SEED = 42
    VERBOSE = True
    
    args = create_parser().parse_args() 

    # Load the Amazon data
    feedback = load_data(args.dataset).load_feedback()
    features, item_ids = load_data(args.dataset).load_visual_feature()
    
    # Visual features
    item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)

    # Evaluation and train/test splits.
    ratio_split = RatioSplit(
        data=feedback,
        test_size=0.1, # Test size is not reported on the VBPR paper.
        rating_threshold=0.5,
        exclude_unknowns=True,
        verbose=VERBOSE,
        item_image=item_image_modality,
        seed = SEED,
        )

    # Instantiate models using similar parameters to the VBPR paper.
    models = [MostPop(),
          BPR(k=10, learning_rate=0.1, seed=SEED), # k are the latent factors 
          VBPR(k=10, k2=10, learning_rate=0.1, seed=SEED, use_gpu=True),
          CausalRec(k=10, k2=10, n_epochs=1, learning_rate=0.01,
                    mean_feat=features.mean(axis=0), tanh=1, use_gpu=True, seed=SEED)
          ]

    # Instantiate evaluation metric
    auc = cornac.metrics.AUC()

    # Put everything together into an experiment and run it and save the results as a log file.
    cornac.Experiment(eval_method=ratio_split, models=models, metrics=[auc], save_dir='data/output/model_eval').run()


if __name__ == '__main__':
    main()