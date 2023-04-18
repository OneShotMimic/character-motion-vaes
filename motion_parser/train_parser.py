import math
import gym
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

from models import MotionParser
from environments import *

data_dir = ""
train_label = ""
test_label = ""

class MyDataSet(Data.Dataset):
  def __init__(self, mocap_inputs, control_outputs):
    super(MyDataSet, self).__init__()
    self.mocap_inputs = mocap_inputs
    self.control_outputs = control_outputs
  
  def __len__(self):
    return self.mocap_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.mocap_inputs[idx], self.control_outputs[idx]

def load_data(tag):
    mocap_inputs = torch.load(data_dir+"/x_"+tag+".pt").cuda()
    control_outputs = torch.load(data_dir+"/actor_"+tag+".pt").cuda()
    return mocap_inputs, control_outputs


def make_gym_environment(args):

    pose_vae_path = os.path.join(current_dir, args.vae_path)

    env = gym.make(
        "{}:{}".format(args.env_module, args.env_name),
        num_parallel=args.num_parallel,
        device=args.device,
        pose_vae_path=pose_vae_path,
        rendered=False,
        frame_skip=args.frame_skip,
    )
    env.seed(args.seed)

    return env

def train(model, epochs=50, lr=1e-3, momentum=0.99):
    train_label = train_label
    
    mocap_inputs, control_outputs = load_data(train_label)
    loader = Data.DataLoader(MyDataSet(mocap_inputs, control_outputs), mocap_inputs.size(0))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        optimizer.zero_grad()
        actions = model(mocap_inputs)
        loss = 0
        for i in range(len(mocap_inputs)):
            loss += criterion(actions[i, :], control_outputs[i, :])
        print(loss)
        loss.backward()
        optimizer.step()
        print("epoch: ", epoch+1, "\tloss: ", loss)

def test(model):
    test_label = test_label
    mocap_inputs, control_outputs = load_data(test_label)
    output = model(mocap_inputs)
    torch.save(output, data_dir+"/pred_"+test_label+".pt")

def main():
    lr = 1e-2
    epoch = 100
    model = MotionParser(d_motion=267, d_control=32).cuda()
    print("\n learning rate:", lr, "; ", epoch, " epoch")
    train(model, epochs=epoch, lr=lr)
    test(model)

if __name__ == "__main__":
    main()
