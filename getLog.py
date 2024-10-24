with open("/home/lorenzo/Scrivania/Progetto_Pannone/new2024/resultsNew.txt") as f:
    rows = f.readlines()
    scores = {"jaccard" : [], "loss" : [], "f1" : []}

    for i in rows:
        if i.startswith("Val Jaccard"):
            scores["jaccard"].append(float(i[i.index(": ") + 2: 18]))
        elif i.startswith("Val Loss"):
            scores["loss"].append(float(i[10:15]))
        elif i.startswith("Val F1-Score"):
            scores["f1"].append(float(i[14:19]))

    for i in scores.keys():
        print(i, end= " ")
        for l in scores[i]:
            print(l)