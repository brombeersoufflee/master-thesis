import matplotlib.pyplot as plt
import json

history=[]
for i in range(1,6):
    with open(f'./CNN_aug/history{i}(2).json') as f:
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
plt.savefig("./CNN_aug/auc")

figure = plt.figure(figsize=(9, 4))
for i in range(0,5):
    # plt.plot(history['accuracy'])
    plt.plot(history[i].get('val_f1_score').keys(), history[i].get('val_f1_score').values())
plt.title('5 fold model val f1')
plt.ylabel('f1')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
plt.savefig("./CNN_aug/f1")


# summarize history for loss
figure = plt.figure(figsize=(9, 4))
for i in range(0,5):
    plt.plot( history[i].get('loss').keys(), history[i].get('loss').values())
plt.title('5 fold model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([f'fold {i+1}' for i in range(0,5)], loc='upper left')
plt.savefig("./CNN_aug/losstrain")