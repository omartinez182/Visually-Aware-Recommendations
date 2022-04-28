import cornac
from cornac.datasets import tradesy
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from cornac.models import MostPop, MMMF, BPR, VBPR


def main():
    """
    Experiment to compare the performance of VBPR withthe MostPop,
    MMMF, and BPR Recommenders.
    """
    SEED = 42
    VERBOSE = True

    # Load the tradesy data
    feedback = tradesy.load_feedback()
    features, item_ids = tradesy.load_visual_feature()  # BIG file

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
            MMMF(k=10, learning_rate=1, seed=SEED), # k are the latent factors 
            BPR(k=10, learning_rate=0.1, seed=SEED),
            VBPR(k=10, k2=10, learning_rate=0.1, seed=SEED)]

    # Instantiate evaluation metric
    auc = cornac.metrics.AUC()

    # Put everything together into an experiment and run it and save the results as a log file.
    cornac.Experiment(eval_method=ratio_split, models=models, metrics=[auc], save_dir='data/output/model_eval').run()


if __name__ == "__main__":
    main()