import matplotlib.pyplot as plt
import json

history=[]
for i in range(1,6):
    with open(f'./CNN_model/history{i}.json') as f:
        history.append(json.load(f))

# print(history[1])
# summarize history for accuracy
figure = plt.figure(figsize=(9, 4))
for i in range(0,5):
    # plt.plot(history['accuracy'])
    plt.plot(history[i].get('val_auc').keys(), history[i].get('val_auc').values())
plt.title('5 fold model val auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
plt.savefig("./CNN_model/auc")

# summarize history for loss
figure = plt.figure(figsize=(9, 4))
for i in range(0,5):
    plt.plot( history[i].get('val_loss').keys(), history[i].get('val_loss').values())
plt.title('5 fold model val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
plt.savefig("./CNN_model/loss")