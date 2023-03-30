from reha.Trainer import train

if __name__ == '__main__':
    train(30000, batch_size=64, save_interval=200)