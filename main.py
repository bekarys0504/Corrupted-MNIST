import argparse
import sys

import torch
from torch import nn
from torch import optim

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        LEARNING_RATE = 0.003
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=LEARNING_RATE)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        epochs = 30
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # TODO: Training pass
                optimizer.zero_grad()

                output = model(images)
                
                labels = labels.type(torch.LongTensor)
                
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(train_set)}")

        torch.save(model.state_dict(), 'checkpoint.pth')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        state_dict = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)
        _, test_set = mnist()
        
        images, labels = next(iter(test_set))
        images = torch.reshape(images, (64, -1))
         # Get the class probabilities
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    