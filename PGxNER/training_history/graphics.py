import os
import seaborn as sns 
import matplotlib.pyplot as plt
import re


sns.set_theme()
dirlist = [dir for dir in os.listdir("./") if os.path.isdir(dir)]

for dir in dirlist : 
    train_history_path = os.path.join(dir,"train_loss")
    if os.path.isfile(train_history_path) :
        fig, ax = plt.subplots()
        with open(train_history_path,'r') as f : 
            train_losses = list(map(lambda line : int(re.sub(r"[\W]*", "", line)),f.readlines()))
            ax.plot(train_losses, color = 'g')
            ax.set_xlim([0, len(train_losses)])
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss")
            ax.set_title("Training set cross-entropy loss over batch")
            fig.savefig(os.path.join(dir,"Training_loss.png"))
            plt.close()
    
    if os.path.isfile(os.path.join(dir,"valid_loss")) :
        valid_history_path = os.path.join(dir,"valid_loss")
    elif os.path.isfile(os.path.join(dir,"valid_scores")) :
        valid_history_path = os.path.join(dir,"valid_scores")
    if valid_history_path is not None :
        fig, ax = plt.subplots()
        with open(valid_history_path,'r') as f : 
            valid_scores = [int(re.search(r"'f':\s\d+",line)[0][5:]) for line in f.readlines()]
            epochs = list(range(10,len(valid_scores)+10))
            ax.plot([0]+epochs, [0]+valid_scores, color = 'b')
            ax.set_xlim([0, len(valid_scores)+10])
            ax.set_ylim([0, 100])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title("Valid set F1-scores over epochs")
            best, best_index = max(valid_scores), valid_scores.index(max(valid_scores))
            plt.axhline(y = best, linestyle = '--')
            ax.text(x = best_index, y = best + 5,s = f"best f1 : {best/100} \n on epoch : {best_index+10}", fontsize = 8)
            fig.savefig(os.path.join(dir,"Valid_score.png"))
            plt.close()
    

        