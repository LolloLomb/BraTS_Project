Implementare pesi per la DiceLoss (inverso della frequenza)   
(1, 419, 40, 151)   
Pesi per la FocalLoss con gamma = 2   
(0.001699, 0.686999, 0.064999, 0.2463)   
In fase di allenamento con CUDA impostare 254 workers   

Funziona!   
Posso ora cambiare i pesi delle Loss, effettuare Data Augmentation, modificare il valore di gamma della Focal Loss, posso cambiare ottimizzatore ed implementare un learning rate scheduler oppure posso usare la Tversky Loss (estensione della Dice che bilancia precisione e richiamo)   
Esistono funzioni di perdita che hanno il focus sui contorni (dato che i miei sono esageratamente larghi)   

