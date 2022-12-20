import sys
import argparse
from kdbnet.parsing import add_train_args
from kdbnet.experiment import DTAExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KDBNet experiment')
    add_train_args(parser)
    args = parser.parse_args()    

    exp = DTAExperiment(
        task=args.task,
        split_method=args.split_method,
        contact_cutoff=args.contact_cutoff,
        num_rbf=args.num_rbf,
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        drug_gcn_dims=args.drug_gcn_dims,
        drug_fc_dims=args.drug_fc_dims,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        uncertainty=args.uncertainty,
        parallel=args.parallel,
        output_dir=args.output_dir,
        save_log=args.save_log,
    )

    if args.save_prediction or args.save_log or args.save_checkpoint:
        exp.saver.save_config(args.__dict__, 'args.yaml')

    exp.train(n_epochs=args.n_epochs, patience=args.patience,
        eval_freq=args.eval_freq, test_freq=args.test_freq,
        monitoring_score=args.monitor_metric)

    if args.recalibrate:
        val_results = exp.test(test_loader=exp.task_loader['valid'], test_df=exp.task_df['valid'],
            test_tag="valid set", print_log=True)
    recalib_df = val_results['df'] if args.recalibrate else None

    test_results = exp.test(save_prediction=args.save_prediction, recalib_df=recalib_df,
        test_tag="Ensemble model", print_log=True)
