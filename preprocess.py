import pandas as pd
import numpy as np
import os
import ast
import cv2

def loadPreprocessedImages():
    df = pd.read_csv("./artemis/data/image-emotion-histogram.csv")
    directory = "./images/"
    y_labels = []

    i = 1
    for index, row in df.iterrows():
        print(f"Preprocessing artwork {i}")
        path = f"./wikiart/{row[0]}/{row[1]}.jpg"
        name = str(row[1])
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(os.path.join(directory, f"{name}.jpg"), img)

        y_label = ast.literal_eval(row[2])
        normalized = np.asarray(y_label) / sum(y_label)

        y_labels.append(normalized)
        i += 1

    print(y_labels)
    y_labels = np.asarray(y_labels)
    np.savetxt("labels.txt", y_labels, delimiter=", ")
    
    
def getCaptions():
    # train - 76031, val - 2000, test - 2000
    df = pd.read_csv("./data/artemis_dataset_release_v0.csv")
    captions = defaultdict(list)
    for index, row in df.iterrows():
        if len(captions[row[1]]) < 5:
            captions[row[1]].append(row[3])
    print("Done adding to dict!")
    with open("captions.txt", 'w') as f:
        print(captions, file=f)
        
        
if __name__ == "__main__":
    loadPreprocessedImages()
