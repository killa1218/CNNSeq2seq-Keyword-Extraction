# Install deps
sudo apt-get install -y python3 python3-dev python3-pip python3-zmq ipython ipython-notebook libzmq3-dev libssl-dev

# Install torch
rm -rf ~/torch
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
bash install-deps
./install.sh
cd ~
source .bashrc

# Install iTorch
rm -rf ~/iTorch
git clone https://github.com/facebook/iTorch.git
cd iTorch
luarocks make
cd ~

# Install NCCL
rm -rf nccl
git clone https://github.com/NVIDIA/nccl.git
cd nccl
sudo make install
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc

# Install lua nn
luarocks install nn

# Install Intel MKL
bash /NAS/Share/YidingTian/l_mkl_2017.3.196/install.sh
