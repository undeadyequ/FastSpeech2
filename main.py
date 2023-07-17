import argparse

from evaluation.visualize import *
from IIV.train import initiate_iiv_train, get_trained_iiv_emb
from train import main
import yaml
from IIV.visualize import show_iiv_distance
from config.ESD.constants import dimension_name
from config.ESD.constants import emo_contri_dimN, emo_optimal_clusterN, emo_optimal_clusterN_ver1, emo_optimal_clusterN_test
from preprocess_ESD import prepare_esd, prepare_grp_id, check_contribute_score
def main(start=1, end=2, configs=None):
    base_dir = "/home/rosen/project/FastSpeech2/"
    processed_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/"
    emo_emb_dir = base_dir + "ESD/emo_reps"
    psd_emb_dir = base_dir + "ESD/psd_reps"
    meta_json = base_dir + "ESD/metadata.json"
    idx_emo_dict = base_dir + "ESD/metadata_22322.json"

    model_config, train_config = configs
    iivmodel = model_config["iiv_model"]
    mean_anchor = train_config["iiv_miner"]["mean_anchor"]
    inter_margin = train_config["iiv_miner"]["inter_margin"]
    intra_margin = train_config["iiv_miner"]["intra_margin"]
    inter_weight = train_config["iiv_miner"]["inter_weight"]
    batch_size = train_config["iiv_optimizer"]["batch_size"]
    distance = train_config["iiv_optimizer"]["distance"]
    lossType = train_config["iiv_optimizer"]["lossType"]
    maxPoint = -1

    # out
    model_f = "iiv_{}_anchor{}_{}_{}_w{}.pt".format(iivmodel,
                                                    mean_anchor,
                                                    str(inter_margin).replace(".", ""),
                                                    str(intra_margin).replace(".", ""),
                                                    str(inter_weight).replace(".", ""),
                                                    )
    best_model = os.path.join(base_dir, "IIV/{}".format(model_f))
    saved_ref_embs = processed_dir + "{}".format(model_f.split(".")[0])
    iiv_dstr_png = base_dir + "IIV/exp/{}_{}.png".format(model_f.split(".")[0], maxPoint)

    # preprocess psd
    if start <= 0 and end >= 0:
        pass

    # Prepare psd emb, group id and metadata.json
    if start <= 1 and end >= 1:
        print("Start step {}: Prepare esd dataset".format("1"))
        prepare_esd()
    # Get energy, duration, pitch given wav and textGrid.
    if start <= 2 and end >= 2:
        print("Start step {}: Cluster esd dataset".format("2"))
        out_dir = base_dir + "ESD"
        cluster_N = emo_optimal_clusterN
        out_meta = "metadata_{}.json".format("".join([str(cn) for cn in list(cluster_N.values())]))
        prepare_grp_id(
            data_dir=out_dir,
            psd_extract=False,
            emo_clusterN=emo_optimal_clusterN,
            out_meta=out_meta
        )
        check_contribute_score(psd_emb_dir, meta_json, cluster_N, cluster_N)


    # Train IIV model
    if start <= 3 and end >= 3:
        # Show Intra- and Inter- variation vis after train
        print("Start step {}: train IIV embedding.".format("3"))
        initiate_iiv_train(emo_emb_dir,
                           psd_emb_dir,
                           idx_emo_dict,
                           iivmodel,
                           distance,
                           lossType,
                           choose_anchor=mean_anchor,
                           inter_type_of_triplets="semihard",
                           intra_type_of_triplets="semihard",
                           inter_margin=inter_margin,
                           intra_margin=intra_margin,
                           inter_weight=inter_weight,
                           batch_size=batch_size,
                           model_f=best_model,
                           num_epochs=15
                           )
    # Get IIV embeddings by best iiv models
    if start <= 4 and end >= 4:
        print("Start step {}: save IIV embedding by best model.".format("4"))
        get_trained_iiv_emb(emo_emb_dir, psd_emb_dir, idx_emo_dict, saved_ref_embs, best_model)

    # Visualization
    if start <= 5 and end >= 5:
        print("Start step {}: visualize IIV embedding.".format("5"))
        for name, emb_dir in zip(["iiv"], [saved_ref_embs]):
            show_iiv_distance(
                emb_dir,
                idx_emo_dict,
                iiv_dstr_png,
                maxPoint,
                20,
                show_cluster=True
            )
    # evaluation
    if start <= 6 and end >= 6:
        pass


if __name__ == '__main__':
    config_dir = "/home/rosen/project/FastSpeech2/config/ESD/"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=config_dir + "model_fastspeechIIV.yaml"
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=config_dir + "train.yaml"
    )

    args = parser.parse_args()
    start_step = 3
    end_step = 5

    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = model_config, train_config
    main(start_step, end_step, configs)
