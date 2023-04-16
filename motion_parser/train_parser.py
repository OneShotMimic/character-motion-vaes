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


class CustomAutoDiff:
    def __init__(self, model, window_size=2):
        self.model = model
        self.window_size = window_size
        self.us = []
        self.xs = []
        self.jacobians_u = []
        self.jacobians_x = []
        self.results = []

    def forward(self,x):
        self.xs.append(x)
        with torch.no_grad():
            res = self.model(x)
        res.requires_grad_(True)
        self.results.append(res)
        jac = torch.autograd.functional.jacobian(self.model,(x))
        self.jacobians_x.append(jac[0])
        return res

    def backprop_window(self,index):
        gradient = torch.zeros_like(self.xs[index])
        for i in range(min(index+self.window_size,len(self.results))-1,index-1,-1):
            gradient = torch.matmul(gradient,self.jacobians_x[i]) + self.results[i].grad
        gradient = torch.matmul(gradient, self.jacobians_u[index])
        return gradient

    def backprop(self):
        """
        Assume self.results already have gradients
        """
        for i in range(len(self.us)):
            self.us[i].grad = self.backprop_window(i)

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
    mocap_inputs = torch.load("/home/ericcsr/extra_disk/jrzhu/data/x_"+tag+".pt").cuda()
    control_outputs = torch.load("/home/ericcsr/extra_disk/jrzhu/data/actor_"+tag+".pt").cuda()
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

def train_fake(model, epochs=50, lr=1e-3, momentum=0.99):
    train_label = "400"
    window_size = 10
    
    autodiff = CustomAutoDiff(model=model, window_size=window_size)
    mocap_inputs, control_outputs = load_data(train_label)
    loader = Data.DataLoader(MyDataSet(mocap_inputs, control_outputs), mocap_inputs.size(0))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        optimizer.zero_grad()
        actions = autodiff.forward(mocap_inputs)
        loss = 0
        for i in range(len(mocap_inputs)):
            loss += criterion(actions[i, :], control_outputs[i, :])
        print(loss)
        loss.backward()
        optimizer.step()
        print("epoch: ", epoch+1, "   average loss: ", sum_loss/count)

def train(model, epochs=50, lr=1e-3, momentum=0.99):

    args = SimpleNamespace(
        env_module="environments",
        env_name="TargetEnv-v0",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_parallel=400,
        vae_path="../vae_motion/models/",
        frame_skip=1,
        seed=16,
        load_saved_model=False,
    )

    args.num_parallel *= args.frame_skip
    env = make_gym_environment(args)

    train_label = "400"
    window_size = 40
    
    autodiff = CustomAutoDiff(model=model, window_size=window_size)
    mocap_inputs, control_outputs = load_data(train_label)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        env.reset()
        optimizer.zero_grad()
        sum_loss = 0
        count = 0
        actions = autodiff.forward(mocap_inputs)
        print(actions.size())
        obs = env.get_vae_next_frame(actions).view(len(mocap_inputs), -1)
        print(obs.size())
        loss = 0
        for i in range(len(mocap_inputs)):
            loss += criterion(obs[i, :], mocap_inputs[i, :])
        print(loss)
        sum_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
        print("epoch: ", epoch+1, "   average loss: ", sum_loss/count)

def test(model):
    test_label = "400_2"
    mocap_inputs, control_outputs = load_data(test_label)
    output = model(mocap_inputs)
    torch.save(output, "/home/ericcsr/extra_disk/jrzhu/data/pred_"+test_label+".pt")

def main():
    model = MotionParser(d_motion=267, d_control=32).cuda()
    print("\n learning rate: 1e-2; 100 epoch")
    train_fake(model, epochs=100, lr=1e-2)
    test(model)

if __name__ == "__main__":
    main()
