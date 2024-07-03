from Classes import LSTMbyHand, LightningLSTM
import torch
import lightning as L

from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    # model = LSTMbyHand()
    model = LightningLSTM()

    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])

    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    trainer = L.Trainer(max_epochs = 300)
    # path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
    
    # trainer = L.Trainer(max_epochs = 3000)
    
    trainer.fit(model, train_dataloaders=dataloader)
    # trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)
    
    print("\nNow let's compare the observed and prediced values ...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 0, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

        
