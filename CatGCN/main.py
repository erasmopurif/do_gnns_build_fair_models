import torch

import numpy as np
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergnn import ClusterGNNTrainer

from utils import tab_printer, graph_reader, field_reader, target_reader, label_reader

import neptune.new as neptune
from fairness import Fairness

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting and scoring the model.
    """
    args = parameter_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    field_index = field_reader(args.field_path)
    target = target_reader(args.target_path)
    user_labels = label_reader(args.labels_path)

    '''Instantiate Neptune client and log arguments'''
    neptune_run = neptune.init(
        project="erasmopurif/CatGCN-fairness-user-profiling",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZGRhYTczYi03MjA1LTRjOTEtYjFjMC1kMjg4ZDZmNWY0ZGMifQ==",)
    neptune_run["sys/tags"].add(["fairness_plus", "new_fairness_ted"])
    neptune_run["seed"] = args.seed
    neptune_run["dataset"] = "JD-small" if "jd" in args.edge_path else "Alibaba-small"
    neptune_run["model"] = "CatGCN"
    neptune_run["label"] = args.label
    neptune_run["lr"] = args.learning_rate
    neptune_run["L2"] = args.weight_decay
    neptune_run["dropout"] = args.dropout
    neptune_run["diag_probe"] = args.diag_probe
    neptune_run["nfm_units"] = args.nfm_units
    neptune_run["grn_units"] = args.grn_units
    neptune_run["gnn_hops"] = args.gnn_hops
    neptune_run["gnn_units"] = args.gnn_units
    neptune_run["balance_ratio"] = args.balance_ratio
    # neptune_run["n_epochs"] = args.epochs

    clustering_machine = ClusteringMachine(args, graph, field_index, target)
    clustering_machine.decompose()
    gnn_trainer = ClusterGNNTrainer(args, clustering_machine, neptune_run)
    gnn_trainer.train_val_test()

    # print("Test nodes idx:", clustering_machine.sg_test_nodes[0].cpu().detach().numpy())
    # print("Test nodes idx[0]:", clustering_machine.sg_test_nodes[0].cpu().detach().numpy()[0])
    # print("Test nodes idx TYPE:", type(clustering_machine.sg_test_nodes[0].cpu().detach().numpy()))
    # print("Test nodes targets:", gnn_trainer.targets)
    # print("Test nodes targets TYPE:", type(gnn_trainer.targets))
    # print("Test nodes preds:", gnn_trainer.predictions)
    # print("Test nodes preds TYPE:", type(gnn_trainer.predictions))

    # print(len(clustering_machine.sg_test_nodes[0].cpu().detach().numpy()))
    # print(len(gnn_trainer.targets))
    # print(len(gnn_trainer.predictions))

    ## Compute fairness metrics
    print("Fairness metrics on sensitive attributes '{}':".format(args.sens_attr))
    fair_obj = Fairness(user_labels, clustering_machine.sg_test_nodes[0], gnn_trainer.targets, gnn_trainer.predictions, args.sens_attr, neptune_run)
    fair_obj.statistical_parity()
    fair_obj.equal_opportunity()
    fair_obj.overall_accuracy_equality()
    fair_obj.treatment_equality()

    neptune_run.stop()


if __name__ == "__main__":
    main()
