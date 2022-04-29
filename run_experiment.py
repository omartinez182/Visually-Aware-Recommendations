import argparse
import cornac
from data import amazon_clothing
from cornac.datasets import tradesy
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from cornac.models import MostPop, MMMF, BPR, VBPR, CausalRec


def main(args):
    """
    Experiment to compare the performance of the MostPop, 
    BPR, VBPR, and CausalRec recommenders on either the Tradesy or
    the Amazon Clothing datasets.
    """
    SEED = 42
    VERBOSE = True

    # Load the Amazon data
    feedback = args.dataset.load_feedback()
    features, item_ids = args.dataset.load_visual_feature()
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default=amazon_clothing)
    args = parser.parse_args()
    main(args)