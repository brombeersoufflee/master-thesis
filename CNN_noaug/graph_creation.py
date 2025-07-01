import matplotlib.pyplot as plt
import json

history=[]
for i in range(1,6):
    with open(f'./CNN_noaug/history{i}(3).json') as f:
        history.append(json.load(f))

# summarize history for metrics
for name in ['accuracy', 'val_accuracy', 'auc', 'val_auc','f1_score', 'val_f1_score']:
    figure = plt.figure(figsize=(9, 4))
    for i in range(0,5):
        # plt.plot(history['accuracy'])
        plt.plot(history[i].get(name).keys(), history[i].get(name).values())
    plt.title(f'5 fold models {name}')
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.ylim(0,1)
    plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
    plt.savefig(f"./CNN_noaug/{name}")

# summarize history for losses
for name in ['loss', 'val_loss']:
    figure = plt.figure(figsize=(9, 4))
    for i in range(0,5):
        plt.plot( history[i].get(name).keys(), history[i].get(name).values())
    plt.title(f'5 fold model {name}')
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
    plt.savefig(f"./CNN_noaug/{name}")