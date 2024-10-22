Implementare pesi per la DiceLoss (inverso della frequenza)   
(1, 419, 40, 151)   
Pesi per la FocalLoss con gamma = 2   
(0.001699, 0.686999, 0.064999, 0.2463)   
In fase di allenamento con CUDA impostare 254 workers   

Funziona!   
Posso ora cambiare i pesi delle Loss, effettuare Data Augmentation, modificare il valore di gamma della Focal Loss, posso cambiare ottimizzatore ed implementare un learning rate scheduler oppure posso usare la Tversky Loss (estensione della Dice che bilancia precisione e richiamo)   
Esistono funzioni di perdita che hanno il focus sui contorni (dato che i miei sono esageratamente larghi)   


Sto provando ad implementare la Resnet3D e a scrivere il wrapper in Lightning   
Vorrei poi usare strumenti più avanzati come lo scheduler, scrivere le metriche e testare il funzionamento   
L'ideale sarebbe aggiungere anche un po' di Data Augmentation   

