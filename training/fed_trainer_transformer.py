import logging

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedTransformerTrainer(ModelTrainer):

    def __init__(self, trainer, model):
        super().__init__(model)
        self.model_trainer = trainer
        self.model = model

    def get_model_params(self):
        # logging.info(self.model.cpu())
        # logging.info(self.model.cpu().state_dict())
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, rounds):
        logging.info("Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.model_trainer.train_dl = train_data
        self.model_trainer.train_model(device=device, rounds=rounds) # FedNLP/training/tc_transformer_trainer.py

    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None, round_idx=0):
        logging.info(args.evaluate_during_training_steps)
        if args.evaluate_during_training_steps == 200 or args.evaluate_during_training_steps == 300: # Baseline and Setup
            if round_idx % 10 == 0:
                logging.info(args)
                self.model_trainer.eval_model(device=device)
            return True
        else:
            # if round_idx % 10 == 0:
            # if True:
            logging.info("Current round is %s; Only evaluate on round %s", str(round_idx), str(args.comm_round-2))
            if round_idx == args.comm_round-2 or round_idx % 10 == 0:
                self.model_trainer.eval_model(device=device)

            return True

def str2list(a):
    a = a.split(',')
    l = a[1].split('[')[1].split(']')[:-1][0].split('.')
    l = [int(i) for i in l]
    return l
