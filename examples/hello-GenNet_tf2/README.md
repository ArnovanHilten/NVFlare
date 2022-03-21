# Hello GenNet TensorFlow

Example of using [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train an Gennet neural network classifier for genotype data using federated averaging ( [FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))), [GenNet](https://www.nature.com/articles/s42003-021-02622-z) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.
Install additional requirements:

```
pip3 install torch
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
set_run_number 1
upload_app hello-tf2
deploy_app hello-tf2 all
start_app all
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
