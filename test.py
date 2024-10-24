from step_two import step_two
import matplotlib.pyplot as plt
import torch
import numpy as np
from newResnetLightning import ResNetLightning


def test_sample(model, val_loader, slice_index=64):  # Modifica per accettare un index di slice
    model.eval()  # Imposta il modello in modalità di valutazione

    # Estrai un campione dal val_loader
    for batch in val_loader:
        x, true_mask = batch  # x: input images, true_mask: ground truth masks
        break  # Prendi solo il primo batch

    # Sposta i tensori su GPU se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = x.to(device)
    true_mask = true_mask.permute(0, 4, 1, 2, 3).to(device)  # Cambia l'ordine delle dimensioni e sposta su GPU

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        y_pred = model(x)  # Esegui la previsione
    
    # Rimuovi la codifica one-hot usando argmax
    y_pred_np = y_pred.cpu().numpy()
    true_mask_np = true_mask.cpu().numpy()

    y_pred_argmax = np.argmax(y_pred_np, axis=1)  # Ottieni la maschera predetta
    true_mask_argmax = np.argmax(true_mask_np, axis=1)  # Ottieni la maschera vera

    # Contare i valori nel range 0-3 per le previsioni
    flat_tensor_pred = torch.tensor(y_pred_argmax).flatten()  # Crea un tensore da numpy e appiattisci
    counts_pred = torch.bincount(flat_tensor_pred.long())  # Usa .long() per convertire in tipo Long

    # Stampa il conteggio per le previsioni
    print("Conteggio dei valori 0-3 (Predizioni):")
    for value in range(4):  # Intervallo da 0 a 3
        print(f"Valore {value}: {counts_pred[value].item()} occorrenze")

    # Contare i valori nel range 0-3 per la maschera vera
    flat_tensor_true = torch.tensor(true_mask_argmax).flatten()  # Crea un tensore da numpy e appiattisci
    counts_true = torch.bincount(flat_tensor_true.long())  # Usa .long() per convertire in tipo Long

    # Stampa il conteggio per le maschere vere
    print("Conteggio dei valori 0-3 (Maschera Vera):")
    for value in range(4):  # Intervallo da 0 a 3
        print(f"Valore {value}: {counts_true[value].item()} occorrenze")

    input("Press any key to plot\n")

    # Stampa la slice specificata
    save_slices(y_pred_argmax[0, :, slice_index], true_mask_argmax[0, :, slice_index], "test1")  # Passa la slice desiderata




def save_slices(pred_slice, true_slice, file_name):
    # Crea la figura
    import os
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Visualizza la maschera vera
    axs[0].imshow(true_slice, cmap='gray')
    axs[0].set_title("Maschera Vera")
    axs[0].axis('off')

    # Visualizza la previsione
    axs[1].imshow(pred_slice, cmap='gray')
    axs[1].set_title("Maschera Predetta")
    axs[1].axis('off')

    # Salva la figura nella cartella corrente
    save_path = os.path.join(os.getcwd(), f"{file_name}.png")
    plt.savefig(save_path)

    flat_tensor_true = torch.tensor(pred_slice).flatten()  # Crea un tensore da numpy e appiattisci
    counts_true = torch.bincount(flat_tensor_true.long())  # Usa .long() per convertire in tipo Long
    for value in range(4):  # Intervallo da 0 a 3
        print(f"Valore {value}: {counts_true[value].item()} occorrenze")

    # Chiude la figura per liberare memoria
    plt.close(fig)


def checkpoint_loader(checkpoint_path):
    # Carica il modello dal checkpoint
    model = ResNetLightning.load_from_checkpoint(checkpoint_path, 
                                        in_channels=3, 
                                        out_channels=4)
    model.to("cuda")
    print(f"\nCheckpoint caricato da {checkpoint_path}")
    return model



def main():

    checkpoint_path = "best-model-epoch=30-val_jaccard=0.54.ckpt"
    model = checkpoint_loader(checkpoint_path)

    _, val_loader = step_two(test = True)
    test_sample(model, val_loader, slice_index=64)  # Passa l'indice della slice che desideri visualizzare


main()