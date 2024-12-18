from step_two import step_two
from newResnetLightning import ResNetLightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint # To save training checkpoints
from lightning.pytorch.callbacks import EarlyStopping   # To stop early when val_loss in minimized
from lightning import Trainer                           # To train the model

def main():
    
    train_loader, val_loader = step_two()  # Create training and validation loaders

    model = ResNetLightning(3, 4) # Create an instance of the ResNet3D Model
    print(f'Model parameters device: {next(model.parameters()).device}')


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  
        dirpath='checkpoints/',  
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        mode='min' 
    )

    max_epochs = int(input("How many epochs? "))

    trainer = Trainer(
        max_epochs=max_epochs,  # Set maximum epochs for training
        callbacks=[checkpoint_callback, early_stopping_callback],  # Include the checkpoint callback
        #devices='auto',  # Automatically choose devices (CPU or GPU)
        accelerator='gpu',  # Use CPU for training
        precision='16-mixed'
    )

    trainer.fit(model, train_loader, val_loader)
    


# Entry point of the script
if __name__ == '__main__':
    main() 
