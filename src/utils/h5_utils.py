import json

def save_metrics(epoch, loss, acc, path='metrics.json'):
    entry = {'epoch': epoch, 'loss': loss, 'accuracy': acc}
    with open(path, 'a') as f:
        f.write(json.dumps(entry) + '\n')