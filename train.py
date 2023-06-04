import torch
from src.trainer import CustomTrainer
import json
import os


def train():
    """
    Train the model with configuration loaded from json file
    """

    # Load training parameters
    params = json.load(open('config/config.json', 'r'))
    
    # params_string = json.dumps(params, indent=4, separators=", ")
    # print(params_string)

    # Create CustomTrainer instance with loaded training parameters
    trainer = CustomTrainer(**params)

    # print(trainer.__dict__)

    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.setup_training_data()    

    # Set up training details
    model, optimizer, loss_fn, device = trainer.setup_training()

    # Loop through epochs
    for epoch in range(trainer.EPOCHS):  # epochs
        # Train model
        trainer.epoch_train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            epoch)

        # Calculate validation loss and accuracy score
        trainer.epoch_evaluate(
            test_dataloader,
            model,
            loss_fn,
            device,
            epoch,
            use_checkpoint = True)
        
        if trainer.early_stop():
            print("EARLY STOPING!!!")
            break

        trainer.writer.flush()
        trainer.writer.close()

    print('TRAINING DONE')


if __name__ == '__main__':
    train()
