import yaml
from torch import optim
from model import load_net
import dataload
from metric import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
        return content


def load_data(path, batch_size):
    data = dataload.MyDataset(path)
    train_data, validate_data = dataload.dataset_split(data, 0.8)
    train_load = dataload.dataloader(train_data, batch_size)
    validate_load = dataload.dataloader(validate_data, batch_size)
    return train_load, validate_load


def train(train_loader, device, net, epochs, lr, loss_f, optimizer, tensorboard_path):
    net = net.to(device)  # put model to GPU
    for epoch in range(epochs):
        net.train()  # set train mode
        top1 = AverageMeter()  # metric
        train_loader = tqdm(train_loader)  # convert to tqdm type, convenient to add the output of journal
        train_loss = 0.0
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', lr))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate topk accuracy
            acc1, acc2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)  # batch_size
            # print(n)
            top1.update(acc1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            # tensorboard curve drawing
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()
    print('Finished Training')


def validate(validation_loader, device, model, loss_f):
    model = model.to(device)  # model --> GPU
    model = model.eval()  # set eval mode
    with torch.no_grad():  # network does not update gradient during evaluation
        val_top1 = AverageMeter()
        validate_loader = tqdm(validation_loader)
        validate_loss = 0
        for i, data in enumerate(validate_loader):
            inputs, labels = data[0].to(device), data[1].to(device)  # data, label --> GPU
            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)  # batch_size=32
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validation_loss': '%.6f' % (validate_loss / (i + 1)), 'validation_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc


def main():
    net = load_net("resnet")
    cfg = load_config('config.yaml')

    epochs = cfg['epochs']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    train_path = cfg['train_path']
    tensorboard_path = cfg['tensorboard_path']
    model_save_path = cfg['model_save_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_f = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)

    train_load, validate_load = load_data(train_path, batch_size)
    train(train_load, device, net, epochs, lr, loss_f, optimizer, tensorboard_path)
    val = validate(validate_load, device, net, loss_f)
    print('val_acc: %.2f' % val)
    torch.save(net.state_dict(), model_save_path)

    print(123)


if __name__ == '__main__':
    main()
