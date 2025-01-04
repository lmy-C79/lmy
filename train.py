import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from Vgg16Net import Vgg16_net
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg' 等


def load_dataset(batch_size):
    train_set = torchvision.datasets.CIFAR10(
        root="train", train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = torchvision.datasets.CIFAR10(
        root="test", train=False, download=True, transform=transforms.ToTensor()
    )
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return train_iter, test_iter


def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    global train_acc
    net.train()
    record_train = []
    record_test = []
    record_losses = []  # 新增：用于记录每个epoch的平均损失

    for epoch in range(num_epochs):
        print(f"========== epoch: [{epoch + 1}/{num_epochs}] ==========")
        total, correct, train_loss = 0, 0, 0

        # 使用tqdm包装迭代器
        for i, (X, y) in enumerate(tqdm(train_iter, desc=f"Epoch {epoch + 1}", ncols=80)):
            X, y = X.to(device), y.to(device)
            output = net(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print(
                    f"step: [{i + 1}/{len(train_iter)}], train_loss: {train_loss / (i + 1):.3f} | train_acc: {train_acc:.3f}% | LR: {get_cur_lr(optimizer)}")

        # 每个epoch结束时计算平均损失
        avg_loss = train_loss / len(train_iter)
        record_losses.append(avg_loss)  # 记录平均损失

        if lr_scheduler is not None:
            lr_scheduler.step()

        if test_iter is not None:
            record_test.append(test(net, test_iter, criterion, device))
        record_train.append(train_acc)

    return record_train, record_test, record_losses


def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)

            output = net(X)
            loss = criterion(output, y)

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()

    test_acc = 100.0 * correct / total

    print("test_loss: {:.3f} | test_acc: {:6.3f}%" \
          .format(loss.item(), test_acc))
    net.train()

    return test_acc


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


def plot_loss(record_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(record_losses) + 1), record_losses, label="Average Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 10
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    net = Vgg16_net()
    net = net.to(DEVICE)

    train_iter, test_iter = load_dataset(BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    record_train, record_test, record_losses = train(net, train_iter, criterion, optimizer,
                                                     NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler, test_iter)

    learning_curve(record_train, record_test)
    plot_loss(record_losses)


if __name__ == '__main__':
    main()
