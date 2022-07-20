# Federated Learning with Flower framework

## How to run

**Note: This implementation runs on Mac. Then the device is set "mps". You should change the device according to your hardware.

### Create dataset

This step will create iid or non-iid datasets for clients from MNIST dataset

For example: To reproduce the experiment desribed in the (paper)[].
```bash
python create_dataset.py --type noniid\
                         --data_dir ./datasets/flmnist
                         --num_clients 100\
                         --num_shards 200
```

In this case, the output is a directory **./datasets/flmnist_noniid_100clients/** which stores multiple data partitions of each clients

### Approach 1: Start server-clients manually

Start the server first

```bash
python server.py --rnd 50 --sample_fraction 1.0 --num_clients 2\
                 --batch_size 32 --num_epochs 3 --lr 0.001 --save_dir save_models
```

Then open the other terminals to start each client.
For example, start the client 1.

```bash
python client.py --cid 1\ 
                --data_dir datasets/flmnist_iid_2clients
```

### Approach 2: Start the simulation with many clients

```bash
python simulation.py --sample_fraction 0.1 --num_clients 100 --batch_size 10 \
                    --num_epochs 5 --lr 0.001 --save_dir save_models \
                    --fed_dir datasets/flmnist_noniid_100clients --rnd 50
```

To know more about arguments, please use '--h'.

## Some results

Learning rate of each client is 0.0001
Strategy: FedAvg

| #Clients   | num_epochs   |batch_size   |Data distribution   |Last round   | Best Acc|
|---|---|---|---|---|---|
|2   |3   |32   |iid   | 0.9044  | 0.9702 |
|2   |3   |32   |non iid   | 0.5012 | 0.5635 |
|100   |5  |10   |iid   | 0.8022  | 0.8443 |
|100   |5   |10   |non iid   |  0.4585 | 0.5719 |

## Some resources to start Federated Learning

- Introduction Slides for FL: https://namhoonlee.github.io/courses/optml/s14-fl.pdf
- Video tutorial: https://www.youtube.com/watch?v=nBGQQHPkyNY
- Flower tutorial: https://www.youtube.com/watch?v=Ky6TicaPfVI&t=2656s